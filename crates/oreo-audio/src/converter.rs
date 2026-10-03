use std::collections::VecDeque;

use oreo_core::CancellationToken;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use crate::{AudioError, AudioErrorKind, AudioFormat, AudioLimits, AudioSource, PcmChunk};

const SINC_LENGTH: usize = 128;
const OVERSAMPLING_FACTOR: usize = 128;
const MAX_FLUSH_BLOCKS: usize = 8;

/// Stateful, bounded PCM format conversion for worker threads.
///
/// This type deliberately does not implement or run inside a device callback.
/// It reuses fixed input and output buffers and emits chunks no longer than
/// 100 ms.
pub struct PcmConverter {
    input_format: AudioFormat,
    output_format: AudioFormat,
    input_frames_per_block: usize,
    pending: Vec<f32>,
    output: Vec<f32>,
    resampler: Option<Box<dyn Resampler<f32>>>,
    delay_frames_left: usize,
    total_input_frames: u64,
    emitted_output_frames: u64,
    finished: bool,
}

/// Adapts any bounded source to a fixed downstream PCM format.
///
/// Converted chunks are held only in a bounded transient queue. Pulling the
/// next chunk drives conversion, so capture cannot run ahead of its consumer.
pub struct ConvertingSource<S> {
    source: S,
    converter: PcmConverter,
    queue: VecDeque<PcmChunk>,
    queue_capacity: usize,
    exhausted: bool,
}

impl<S: AudioSource> ConvertingSource<S> {
    /// Wraps `source` and converts it to `output_format` as chunks are pulled.
    ///
    /// # Errors
    ///
    /// Rejects unsupported formats, invalid limits, or converter setup errors.
    pub fn new(
        source: S,
        output_format: AudioFormat,
        limits: AudioLimits,
    ) -> Result<Self, AudioError> {
        let limits = limits.validate()?;
        let converter = PcmConverter::new(source.format(), output_format, limits)?;
        Ok(Self {
            source,
            converter,
            queue: VecDeque::with_capacity(limits.queue_capacity),
            queue_capacity: limits.queue_capacity,
            exhausted: false,
        })
    }

    #[must_use]
    pub fn into_inner(self) -> S {
        self.source
    }
}

impl<S: AudioSource> AudioSource for ConvertingSource<S> {
    fn format(&self) -> AudioFormat {
        self.converter.output_format()
    }

    fn next_chunk(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Option<PcmChunk>, AudioError> {
        if cancellation.is_cancelled() {
            self.queue.clear();
            return Err(AudioError::new(
                AudioErrorKind::Cancelled,
                "audio conversion was cancelled",
            ));
        }
        if let Some(chunk) = self.queue.pop_front() {
            return Ok(Some(chunk));
        }
        if self.exhausted {
            return Ok(None);
        }

        loop {
            if let Some(chunk) = self.source.next_chunk(cancellation)? {
                let queue = &mut self.queue;
                let capacity = self.queue_capacity;
                self.converter.push(&chunk, cancellation, &mut |chunk| {
                    if queue.len() == capacity {
                        return Err(AudioError::new(
                            AudioErrorKind::Capacity,
                            "converted audio queue is full",
                        ));
                    }
                    queue.push_back(chunk);
                    Ok(())
                })?;
                if let Some(chunk) = self.queue.pop_front() {
                    return Ok(Some(chunk));
                }
                continue;
            }

            let queue = &mut self.queue;
            let capacity = self.queue_capacity;
            self.converter.finish(cancellation, &mut |chunk| {
                if queue.len() == capacity {
                    return Err(AudioError::new(
                        AudioErrorKind::Capacity,
                        "converted audio queue is full",
                    ));
                }
                queue.push_back(chunk);
                Ok(())
            })?;
            self.exhausted = true;
            return Ok(self.queue.pop_front());
        }
    }
}

impl PcmConverter {
    /// Creates a converter whose internal block duration follows `limits`.
    ///
    /// # Errors
    ///
    /// Rejects invalid formats/limits or a resampler construction failure.
    pub fn new(
        input_format: AudioFormat,
        output_format: AudioFormat,
        limits: AudioLimits,
    ) -> Result<Self, AudioError> {
        let input_format = input_format.validate()?;
        let output_format = output_format.validate()?;
        let limits = limits.validate()?;
        let input_frames_per_block = limits
            .samples_per_chunk(input_format)?
            .checked_div(usize::from(input_format.channels))
            .ok_or_else(invalid_conversion)?;
        let channels = usize::from(output_format.channels);

        let (resampler, output_frames, delay_frames_left) =
            if input_format.sample_rate_hz == output_format.sample_rate_hz {
                (None, input_frames_per_block, 0)
            } else {
                let ratio = f64::from(output_format.sample_rate_hz)
                    / f64::from(input_format.sample_rate_hz);
                let parameters =
                    SincInterpolationParameters::new(SINC_LENGTH, WindowFunction::Blackman2)
                        .oversampling_factor(OVERSAMPLING_FACTOR)
                        .interpolation(SincInterpolationType::Linear);
                let resampler = Async::<f32>::new_sinc(
                    ratio,
                    1.0,
                    &parameters,
                    input_frames_per_block,
                    channels,
                    FixedAsync::Input,
                )
                .map_err(|_| {
                    AudioError::new(AudioErrorKind::InvalidConfig, "audio conversion is invalid")
                })?;
                let output_frames = resampler.output_frames_max();
                let delay = resampler.output_delay();
                (
                    Some(Box::new(resampler) as Box<dyn Resampler<f32>>),
                    output_frames,
                    delay,
                )
            };

        Ok(Self {
            input_format,
            output_format,
            input_frames_per_block,
            pending: Vec::with_capacity(input_frames_per_block * channels),
            output: vec![0.0; output_frames * channels],
            resampler,
            delay_frames_left,
            total_input_frames: 0,
            emitted_output_frames: 0,
            finished: false,
        })
    }

    #[must_use]
    pub const fn input_format(&self) -> AudioFormat {
        self.input_format
    }

    #[must_use]
    pub const fn output_format(&self) -> AudioFormat {
        self.output_format
    }

    /// Converts one transient chunk and emits zero or more bounded chunks.
    ///
    /// # Errors
    ///
    /// Rejects cancellation, format mismatches, use after finish, or backend
    /// conversion failures.
    pub fn push(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
        emit: &mut dyn FnMut(PcmChunk) -> Result<(), AudioError>,
    ) -> Result<(), AudioError> {
        self.check_active(cancellation)?;
        if chunk.format() != self.input_format {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "audio chunk format does not match converter",
            ));
        }

        let input_channels = usize::from(self.input_format.channels);
        for frame in chunk.samples().chunks_exact(input_channels) {
            self.push_frame(frame);
            self.total_input_frames = self.total_input_frames.saturating_add(1);
            if self.pending.len()
                == self.input_frames_per_block * usize::from(self.output_format.channels)
            {
                if self.resampler.is_some() {
                    self.process_block(None, None, emit)?;
                } else {
                    self.emit_pending(self.input_frames_per_block, emit)?;
                }
            }
        }
        Ok(())
    }

    /// Flushes the final partial block and trims resampler padding to the exact
    /// duration implied by all accepted input frames.
    ///
    /// # Errors
    ///
    /// Returns a redacted cancellation, state, or backend error.
    pub fn finish(
        &mut self,
        cancellation: &CancellationToken,
        emit: &mut dyn FnMut(PcmChunk) -> Result<(), AudioError>,
    ) -> Result<(), AudioError> {
        self.check_active(cancellation)?;
        self.finished = true;

        let expected_frames = self.expected_output_frames();
        if self.resampler.is_none() {
            if !self.pending.is_empty() {
                self.emit_pending(usize::try_from(expected_frames).unwrap_or(usize::MAX), emit)?;
            }
            return Ok(());
        }

        if !self.pending.is_empty() {
            let channels = usize::from(self.output_format.channels);
            let partial_frames = self.pending.len() / channels;
            self.process_block(Some(partial_frames), Some(expected_frames), emit)?;
        }

        let mut flushes = 0;
        while self.emitted_output_frames < expected_frames && flushes < MAX_FLUSH_BLOCKS {
            self.process_block(Some(0), Some(expected_frames), emit)?;
            flushes += 1;
        }
        if self.emitted_output_frames != expected_frames {
            return Err(AudioError::new(
                AudioErrorKind::Backend,
                "audio conversion could not be finalized",
            ));
        }
        Ok(())
    }

    fn check_active(&self, cancellation: &CancellationToken) -> Result<(), AudioError> {
        if cancellation.is_cancelled() {
            return Err(AudioError::new(
                AudioErrorKind::Cancelled,
                "audio conversion was cancelled",
            ));
        }
        if self.finished {
            return Err(AudioError::new(
                AudioErrorKind::InvalidTransition,
                "audio converter is already finished",
            ));
        }
        Ok(())
    }

    fn push_frame(&mut self, frame: &[i16]) {
        match (self.input_format.channels, self.output_format.channels) {
            (1, 1) => self.pending.push(to_float(frame[0])),
            (1, 2) => {
                let sample = to_float(frame[0]);
                self.pending.extend_from_slice(&[sample, sample]);
            }
            (2, 1) => {
                let mixed = i16::midpoint(frame[0], frame[1]);
                self.pending.push(to_float(mixed));
            }
            (2, 2) => self.pending.extend(frame.iter().copied().map(to_float)),
            _ => unreachable!("validated formats only allow one or two channels"),
        }
    }

    fn process_block(
        &mut self,
        partial_frames: Option<usize>,
        output_limit: Option<u64>,
        emit: &mut dyn FnMut(PcmChunk) -> Result<(), AudioError>,
    ) -> Result<(), AudioError> {
        let channels = usize::from(self.output_format.channels);
        let input_frames = self.pending.len() / channels;
        let input = InterleavedSlice::new(&self.pending, channels, input_frames)
            .map_err(|_| AudioError::new(AudioErrorKind::Backend, "audio conversion failed"))?;
        let output_frames = self.output.len() / channels;
        let mut output = InterleavedSlice::new_mut(&mut self.output, channels, output_frames)
            .map_err(|_| AudioError::new(AudioErrorKind::Backend, "audio conversion failed"))?;
        let indexing = partial_frames.map(|frames| Indexing::new().partial_len(frames));
        let (_, written) = self
            .resampler
            .as_mut()
            .expect("resampler exists for process_block")
            .process_into_buffer(&input, &mut output, indexing.as_ref())
            .map_err(|_| AudioError::new(AudioErrorKind::Backend, "audio conversion failed"))?;
        self.pending.clear();

        let skip = self.delay_frames_left.min(written);
        self.delay_frames_left -= skip;
        let available = written - skip;
        let allowed = output_limit.map_or(available, |limit| {
            usize::try_from(limit.saturating_sub(self.emitted_output_frames))
                .unwrap_or(usize::MAX)
                .min(available)
        });
        let first = skip * channels;
        let last = first + allowed * channels;
        let converted = self.output[first..last]
            .iter()
            .copied()
            .map(to_i16)
            .collect::<Vec<_>>();
        self.emit_samples(&converted, emit)
    }

    fn emit_pending(
        &mut self,
        max_frames: usize,
        emit: &mut dyn FnMut(PcmChunk) -> Result<(), AudioError>,
    ) -> Result<(), AudioError> {
        let channels = usize::from(self.output_format.channels);
        let samples = (max_frames * channels).min(self.pending.len());
        let converted = self.pending[..samples]
            .iter()
            .copied()
            .map(to_i16)
            .collect::<Vec<_>>();
        self.pending.clear();
        self.emit_samples(&converted, emit)
    }

    fn emit_samples(
        &mut self,
        samples: &[i16],
        emit: &mut dyn FnMut(PcmChunk) -> Result<(), AudioError>,
    ) -> Result<(), AudioError> {
        let channels = usize::from(self.output_format.channels);
        let max_samples = usize::try_from(self.output_format.sample_rate_hz)
            .ok()
            .and_then(|rate| rate.checked_div(10))
            .and_then(|frames| frames.checked_mul(channels))
            .ok_or_else(invalid_conversion)?;
        for samples in samples.chunks(max_samples) {
            if samples.is_empty() {
                continue;
            }
            emit(PcmChunk::new(self.output_format, samples.to_vec())?)?;
            self.emitted_output_frames = self
                .emitted_output_frames
                .saturating_add(u64::try_from(samples.len() / channels).unwrap_or(u64::MAX));
        }
        Ok(())
    }

    fn expected_output_frames(&self) -> u64 {
        let numerator = self
            .total_input_frames
            .saturating_mul(u64::from(self.output_format.sample_rate_hz));
        let denominator = u64::from(self.input_format.sample_rate_hz);
        numerator.saturating_add(denominator / 2) / denominator
    }
}

fn invalid_conversion() -> AudioError {
    AudioError::new(AudioErrorKind::InvalidConfig, "audio conversion is invalid")
}

fn to_float(sample: i16) -> f32 {
    f32::from(sample) / 32_768.0
}

#[allow(clippy::cast_possible_truncation)]
fn to_i16(sample: f32) -> i16 {
    let scaled = sample.clamp(-1.0, 1.0) * 32_768.0;
    scaled
        .round()
        .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::PcmConverter;
    use crate::{
        AudioError, AudioErrorKind, AudioFormat, AudioLimits, AudioSource, ConvertingSource,
        PcmChunk, STT_FORMAT,
    };
    use oreo_core::CancellationToken;

    #[test]
    fn downmixes_and_resamples_laptop_audio_to_exact_stt_duration() {
        let input = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };
        let mut converter =
            PcmConverter::new(input, STT_FORMAT, AudioLimits::sbc()).expect("valid converter");
        let cancellation = CancellationToken::new();
        let mut output = Vec::new();

        for block in 0..50 {
            let mut samples = Vec::with_capacity(1_920);
            for offset in 0..960 {
                let frame = block * 960 + offset;
                let sample = if frame % 109 < 55 { 12_000 } else { -12_000 };
                samples.extend_from_slice(&[sample, sample]);
            }
            let chunk = PcmChunk::new(input, samples).expect("valid input");
            converter
                .push(&chunk, &cancellation, &mut |chunk| {
                    assert!(chunk.samples().len() <= 1_600);
                    output.extend_from_slice(chunk.samples());
                    Ok(())
                })
                .expect("conversion succeeds");
        }
        converter
            .finish(&cancellation, &mut |chunk| {
                output.extend_from_slice(chunk.samples());
                Ok(())
            })
            .expect("flush succeeds");

        assert_eq!(output.len(), 16_000);
        assert!(output.iter().any(|sample| sample.unsigned_abs() > 5_000));
    }

    #[test]
    fn converts_channels_without_resampling() {
        let mono = AudioFormat {
            sample_rate_hz: 16_000,
            channels: 1,
        };
        let stereo = AudioFormat {
            sample_rate_hz: 16_000,
            channels: 2,
        };
        let cancellation = CancellationToken::new();
        let mut converter =
            PcmConverter::new(mono, stereo, AudioLimits::sbc()).expect("valid converter");
        let mut input_samples = vec![100, -200, 300];
        input_samples.extend(std::iter::repeat_n(400, 317));
        let chunk = PcmChunk::new(mono, input_samples).expect("valid chunk");
        let mut output = Vec::new();
        converter
            .push(&chunk, &cancellation, &mut |chunk| {
                output.extend_from_slice(chunk.samples());
                Ok(())
            })
            .expect("push succeeds");
        converter
            .finish(&cancellation, &mut |chunk| {
                output.extend_from_slice(chunk.samples());
                Ok(())
            })
            .expect("finish succeeds");
        assert_eq!(&output[..6], [100, 100, -200, -200, 300, 300]);
        assert_eq!(output.len(), 640);
    }

    #[test]
    fn rejects_mismatches_cancellation_and_reuse() {
        let input = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 1,
        };
        let cancellation = CancellationToken::new();
        let mut converter =
            PcmConverter::new(input, STT_FORMAT, AudioLimits::sbc()).expect("valid converter");
        let wrong = PcmChunk::new(STT_FORMAT, vec![0; 320]).expect("valid chunk");
        let error = converter
            .push(&wrong, &cancellation, &mut |_| Ok(()))
            .expect_err("format mismatch");
        assert_eq!(error.kind, AudioErrorKind::UnsupportedFormat);

        cancellation.cancel();
        let input_chunk = PcmChunk::new(input, vec![0; 960]).expect("valid chunk");
        let error = converter
            .push(&input_chunk, &cancellation, &mut |_| Ok(()))
            .expect_err("cancelled");
        assert_eq!(error.kind, AudioErrorKind::Cancelled);

        let cancellation = CancellationToken::new();
        converter
            .finish(&cancellation, &mut |_| Ok(()))
            .expect("finish succeeds");
        let error = converter
            .finish(&cancellation, &mut |_| Ok(()))
            .expect_err("second finish fails");
        assert_eq!(error.kind, AudioErrorKind::InvalidTransition);
    }

    #[test]
    fn flushes_a_partial_upsampling_block_to_exact_duration() {
        let input = AudioFormat {
            sample_rate_hz: 8_000,
            channels: 1,
        };
        let output_format = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };
        let cancellation = CancellationToken::new();
        let mut converter =
            PcmConverter::new(input, output_format, AudioLimits::sbc()).expect("valid converter");
        let chunk = PcmChunk::new(input, vec![1_000; 17]).expect("valid chunk");
        let mut output = Vec::new();
        converter
            .push(&chunk, &cancellation, &mut |chunk| {
                output.extend_from_slice(chunk.samples());
                Ok(())
            })
            .expect("push succeeds");
        converter
            .finish(&cancellation, &mut |chunk| {
                output.extend_from_slice(chunk.samples());
                Ok(())
            })
            .expect("finish succeeds");

        assert_eq!(output.len(), 17 * 6 * 2);
        assert!(output.chunks_exact(2).all(|frame| frame[0] == frame[1]));
    }

    struct TestSource {
        format: AudioFormat,
        chunks: VecDeque<PcmChunk>,
    }

    impl AudioSource for TestSource {
        fn format(&self) -> AudioFormat {
            self.format
        }

        fn next_chunk(
            &mut self,
            cancellation: &CancellationToken,
        ) -> Result<Option<PcmChunk>, AudioError> {
            if cancellation.is_cancelled() {
                return Err(AudioError::new(
                    AudioErrorKind::Cancelled,
                    "fixture was cancelled",
                ));
            }
            Ok(self.chunks.pop_front())
        }
    }

    #[test]
    fn converting_source_bridges_device_audio_to_stt_format() {
        let input = AudioFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        };
        let source = TestSource {
            format: input,
            chunks: [
                PcmChunk::new(input, vec![2_000; 1_920]).expect("valid chunk"),
                PcmChunk::new(input, vec![-2_000; 960]).expect("valid partial chunk"),
            ]
            .into(),
        };
        let mut source =
            ConvertingSource::new(source, STT_FORMAT, AudioLimits::sbc()).expect("valid source");
        let cancellation = CancellationToken::new();
        let mut samples = 0;
        while let Some(chunk) = source
            .next_chunk(&cancellation)
            .expect("conversion succeeds")
        {
            assert_eq!(chunk.format(), STT_FORMAT);
            samples += chunk.samples().len();
        }
        assert_eq!(samples, 480);
        assert!(
            source
                .next_chunk(&cancellation)
                .expect("exhausted source remains stable")
                .is_none()
        );
    }

    #[test]
    fn converting_source_discards_queued_audio_on_cancellation() {
        let format = STT_FORMAT;
        let source = TestSource {
            format,
            chunks: [PcmChunk::new(format, vec![1; 640]).expect("valid chunk")].into(),
        };
        let mut limits = AudioLimits::sbc();
        limits.queue_capacity = 2;
        let mut source =
            ConvertingSource::new(source, format, limits).expect("valid converting source");
        let cancellation = CancellationToken::new();
        let first = source
            .next_chunk(&cancellation)
            .expect("first chunk converts")
            .expect("first chunk exists");
        assert_eq!(first.samples().len(), 320);

        cancellation.cancel();
        let error = source
            .next_chunk(&cancellation)
            .expect_err("queued audio is not returned after cancellation");
        assert_eq!(error.kind, AudioErrorKind::Cancelled);
    }
}
