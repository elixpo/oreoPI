use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, TrySendError, sync_channel};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, FromSample, I24, Sample, SampleFormat, SizedSample, Stream, StreamConfig, U24};
use oreo_core::CancellationToken;

use crate::{
    AudioError, AudioErrorKind, AudioFormat, AudioLimits, AudioOutput, AudioSource, PcmChunk,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AudioDeviceSummary {
    pub host: String,
    pub default_input: Option<String>,
    pub default_output: Option<String>,
}

/// Returns non-secret names for the host's default input and output devices.
///
/// # Errors
///
/// Returns a redacted backend error when the audio host cannot be queried.
pub fn default_audio_devices() -> Result<AudioDeviceSummary, AudioError> {
    let host = cpal::default_host();
    Ok(AudioDeviceSummary {
        host: host.id().name().to_owned(),
        default_input: host.default_input_device().map(|device| device.to_string()),
        default_output: host
            .default_output_device()
            .map(|device| device.to_string()),
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioIoSnapshot {
    pub captured_chunks: u64,
    pub dropped_chunks: u64,
    pub played_samples: u64,
    pub queued_samples: u64,
    pub underrun_callbacks: u64,
    pub stream_errors: u64,
}

#[derive(Default)]
struct IoCounters {
    captured_chunks: AtomicU64,
    dropped_chunks: AtomicU64,
    played_samples: AtomicU64,
    queued_samples: AtomicU64,
    underrun_callbacks: AtomicU64,
    stream_errors: AtomicU64,
}

impl IoCounters {
    fn snapshot(&self) -> AudioIoSnapshot {
        AudioIoSnapshot {
            captured_chunks: self.captured_chunks.load(Ordering::Relaxed),
            dropped_chunks: self.dropped_chunks.load(Ordering::Relaxed),
            played_samples: self.played_samples.load(Ordering::Relaxed),
            queued_samples: self.queued_samples.load(Ordering::Acquire),
            underrun_callbacks: self.underrun_callbacks.load(Ordering::Relaxed),
            stream_errors: self.stream_errors.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone)]
pub struct CaptureControl(Arc<AtomicBool>);

impl CaptureControl {
    pub fn stop(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

pub struct CpalInputSource {
    format: AudioFormat,
    receiver: Receiver<Vec<i16>>,
    recycle_sender: SyncSender<Vec<i16>>,
    control: CaptureControl,
    faulted: Arc<AtomicBool>,
    counters: Arc<IoCounters>,
    _stream: Stream,
}

impl CpalInputSource {
    /// Opens and starts the default microphone using its native PCM format.
    ///
    /// # Errors
    ///
    /// Returns a redacted device, format, configuration, or stream failure.
    pub fn open_default(limits: AudioLimits) -> Result<Self, AudioError> {
        let limits = limits.validate()?;
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or_else(device_error)?;
        let supported = device.default_input_config().map_err(|_| device_error())?;
        let format = AudioFormat {
            sample_rate_hz: supported.sample_rate(),
            channels: supported.channels(),
        }
        .validate()?;
        let samples_per_chunk = limits.samples_per_chunk(format)?;
        let (sender, receiver) = sync_channel(limits.queue_capacity);
        let (recycle_sender, recycle_receiver) = sync_channel(limits.queue_capacity);
        for _ in 0..limits.queue_capacity {
            recycle_sender
                .try_send(Vec::with_capacity(samples_per_chunk))
                .map_err(|_| {
                    AudioError::new(
                        AudioErrorKind::InvalidConfig,
                        "audio buffer pool could not be initialized",
                    )
                })?;
        }
        let control = CaptureControl(Arc::new(AtomicBool::new(false)));
        let faulted = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(IoCounters::default());
        let sample_format = supported.sample_format();
        let config = supported.config();
        let stream = build_input_stream(
            &device,
            &config,
            sample_format,
            sender,
            recycle_sender.clone(),
            recycle_receiver,
            samples_per_chunk,
            &control,
            &faulted,
            &counters,
        )?;
        stream.play().map_err(|_| device_error())?;
        Ok(Self {
            format,
            receiver,
            recycle_sender,
            control,
            faulted,
            counters,
            _stream: stream,
        })
    }

    #[must_use]
    pub fn control(&self) -> CaptureControl {
        self.control.clone()
    }

    #[must_use]
    pub fn stats(&self) -> AudioIoSnapshot {
        self.counters.snapshot()
    }
}

impl AudioSource for CpalInputSource {
    fn format(&self) -> AudioFormat {
        self.format
    }

    fn next_chunk(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Option<PcmChunk>, AudioError> {
        loop {
            if cancellation.is_cancelled() {
                self.control.stop();
                return Err(AudioError::new(
                    AudioErrorKind::Cancelled,
                    "audio capture was cancelled",
                ));
            }
            match self.receiver.recv_timeout(Duration::from_millis(20)) {
                Ok(mut samples) => {
                    let chunk = PcmChunk::new(self.format, samples.clone())?;
                    samples.clear();
                    let _ = self.recycle_sender.try_send(samples);
                    return Ok(Some(chunk));
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if self.faulted.load(Ordering::Acquire) {
                        return Err(device_error());
                    }
                    if self.control.is_stopped() {
                        return Ok(None);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    return if self.control.is_stopped() {
                        Ok(None)
                    } else {
                        Err(device_error())
                    };
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    sender: SyncSender<Vec<i16>>,
    recycle_sender: SyncSender<Vec<i16>>,
    recycle_receiver: Receiver<Vec<i16>>,
    samples_per_chunk: usize,
    control: &CaptureControl,
    faulted: &Arc<AtomicBool>,
    counters: &Arc<IoCounters>,
) -> Result<Stream, AudioError> {
    macro_rules! build {
        ($sample:ty) => {
            build_typed_input::<$sample>(
                device,
                config,
                sender,
                recycle_sender,
                recycle_receiver,
                samples_per_chunk,
                control,
                faulted,
                counters,
            )
        };
    }
    match sample_format {
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I24 => build!(I24),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U24 => build!(U24),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        _ => Err(AudioError::new(
            AudioErrorKind::UnsupportedFormat,
            "microphone sample format is unsupported",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_typed_input<T>(
    device: &Device,
    config: &StreamConfig,
    sender: SyncSender<Vec<i16>>,
    recycle_sender: SyncSender<Vec<i16>>,
    recycle_receiver: Receiver<Vec<i16>>,
    samples_per_chunk: usize,
    control: &CaptureControl,
    faulted: &Arc<AtomicBool>,
    counters: &Arc<IoCounters>,
) -> Result<Stream, AudioError>
where
    T: SizedSample + Copy,
    i16: FromSample<T>,
{
    let stopped = control.0.clone();
    let callback_counters = counters.clone();
    let mut assembler = FrameAssembler::new(
        sender,
        recycle_sender,
        recycle_receiver,
        samples_per_chunk,
        callback_counters.clone(),
    );
    let error_faulted = faulted.clone();
    let error_counters = counters.clone();
    device
        .build_input_stream(
            *config,
            move |input: &[T], _| {
                if !stopped.load(Ordering::Acquire) {
                    assembler.push(input.iter().copied().map(i16::from_sample));
                }
            },
            move |_| {
                error_counters.stream_errors.fetch_add(1, Ordering::Relaxed);
                error_faulted.store(true, Ordering::Release);
            },
            Some(Duration::from_secs(2)),
        )
        .map_err(|_| device_error())
}

struct FrameAssembler {
    sender: SyncSender<Vec<i16>>,
    recycle_sender: SyncSender<Vec<i16>>,
    recycle_receiver: Receiver<Vec<i16>>,
    samples_per_chunk: usize,
    pending: Vec<i16>,
    counters: Arc<IoCounters>,
}

impl FrameAssembler {
    fn new(
        sender: SyncSender<Vec<i16>>,
        recycle_sender: SyncSender<Vec<i16>>,
        recycle_receiver: Receiver<Vec<i16>>,
        samples_per_chunk: usize,
        counters: Arc<IoCounters>,
    ) -> Self {
        Self {
            sender,
            recycle_sender,
            recycle_receiver,
            samples_per_chunk,
            pending: Vec::with_capacity(samples_per_chunk),
            counters,
        }
    }

    fn push(&mut self, samples: impl Iterator<Item = i16>) {
        for sample in samples {
            self.pending.push(sample);
            if self.pending.len() == self.samples_per_chunk {
                let Ok(replacement) = self.recycle_receiver.try_recv() else {
                    self.pending.clear();
                    self.counters.dropped_chunks.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
                let chunk = std::mem::replace(&mut self.pending, replacement);
                if let Err(error) = self.sender.try_send(chunk) {
                    let mut rejected = match error {
                        TrySendError::Full(rejected) | TrySendError::Disconnected(rejected) => {
                            rejected
                        }
                    };
                    rejected.clear();
                    let replacement = std::mem::replace(&mut self.pending, rejected);
                    let _ = self.recycle_sender.try_send(replacement);
                    self.counters.dropped_chunks.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.counters
                        .captured_chunks
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

pub struct CpalOutput {
    device: Device,
    native_format: AudioFormat,
    sample_format: SampleFormat,
    config: StreamConfig,
    queue_capacity: usize,
    max_chunk_samples: usize,
    sender: Option<SyncSender<Vec<i16>>>,
    recycle_sender: Option<SyncSender<Vec<i16>>>,
    recycle_receiver: Option<Receiver<Vec<i16>>>,
    stream: Option<Stream>,
    counters: Arc<IoCounters>,
}

impl CpalOutput {
    /// Opens the default speaker without starting playback.
    ///
    /// # Errors
    ///
    /// Returns a redacted device, format, or configuration failure.
    pub fn open_default(limits: AudioLimits) -> Result<Self, AudioError> {
        let limits = limits.validate()?;
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or_else(device_error)?;
        let supported = device.default_output_config().map_err(|_| device_error())?;
        let native_format = AudioFormat {
            sample_rate_hz: supported.sample_rate(),
            channels: supported.channels(),
        }
        .validate()?;
        let max_chunk_samples = usize::try_from(native_format.sample_rate_hz)
            .ok()
            .and_then(|rate| rate.checked_div(10))
            .and_then(|frames| frames.checked_mul(usize::from(native_format.channels)))
            .ok_or_else(|| {
                AudioError::new(AudioErrorKind::InvalidConfig, "speaker buffer is invalid")
            })?;
        Ok(Self {
            device,
            native_format,
            sample_format: supported.sample_format(),
            config: supported.config(),
            queue_capacity: limits.queue_capacity,
            max_chunk_samples,
            sender: None,
            recycle_sender: None,
            recycle_receiver: None,
            stream: None,
            counters: Arc::new(IoCounters::default()),
        })
    }

    #[must_use]
    pub const fn native_format(&self) -> AudioFormat {
        self.native_format
    }

    #[must_use]
    pub fn stats(&self) -> AudioIoSnapshot {
        self.counters.snapshot()
    }
}

impl AudioOutput for CpalOutput {
    fn begin(&mut self, format: AudioFormat) -> Result<(), AudioError> {
        if format.validate()? != self.native_format {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "speaker requires its native audio format",
            ));
        }
        self.stop();
        let (sender, receiver) = sync_channel(self.queue_capacity);
        let pool_capacity = self.queue_capacity.checked_add(1).ok_or_else(|| {
            AudioError::new(AudioErrorKind::InvalidConfig, "speaker pool is invalid")
        })?;
        let (recycle_sender, recycle_receiver) = sync_channel(pool_capacity);
        for _ in 0..pool_capacity {
            recycle_sender
                .try_send(Vec::with_capacity(self.max_chunk_samples))
                .map_err(|_| {
                    AudioError::new(
                        AudioErrorKind::InvalidConfig,
                        "speaker buffer pool could not be initialized",
                    )
                })?;
        }
        let stream = build_output_stream(
            &self.device,
            &self.config,
            self.sample_format,
            receiver,
            recycle_sender.clone(),
            &self.counters,
        )?;
        stream.play().map_err(|_| device_error())?;
        self.sender = Some(sender);
        self.recycle_sender = Some(recycle_sender);
        self.recycle_receiver = Some(recycle_receiver);
        self.stream = Some(stream);
        Ok(())
    }

    fn write(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<(), AudioError> {
        if cancellation.is_cancelled() {
            self.stop();
            return Err(AudioError::new(
                AudioErrorKind::Cancelled,
                "audio playback was cancelled",
            ));
        }
        if chunk.format() != self.native_format {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "speaker audio format changed",
            ));
        }
        let mut samples = self
            .recycle_receiver
            .as_ref()
            .ok_or_else(|| AudioError::new(AudioErrorKind::Backend, "speaker is not started"))?
            .try_recv()
            .map_err(|_| AudioError::new(AudioErrorKind::Capacity, "speaker queue is full"))?;
        samples.extend_from_slice(chunk.samples());
        let sample_count = u64::try_from(samples.len()).unwrap_or(u64::MAX);
        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| AudioError::new(AudioErrorKind::Backend, "speaker is not started"))?;
        self.counters
            .queued_samples
            .fetch_add(sample_count, Ordering::AcqRel);
        if let Err(error) = sender.try_send(samples) {
            subtract_queued(&self.counters.queued_samples, sample_count);
            let (mut rejected, failure) = match error {
                TrySendError::Full(rejected) => (
                    rejected,
                    AudioError::new(AudioErrorKind::Capacity, "speaker queue is full"),
                ),
                TrySendError::Disconnected(rejected) => (rejected, device_error()),
            };
            rejected.clear();
            if let Some(recycle) = &self.recycle_sender {
                let _ = recycle.try_send(rejected);
            }
            return Err(failure);
        }
        Ok(())
    }

    fn stop(&mut self) {
        self.sender = None;
        self.stream = None;
        self.recycle_receiver = None;
        self.recycle_sender = None;
        self.counters.queued_samples.store(0, Ordering::Release);
    }
}

fn build_output_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    receiver: Receiver<Vec<i16>>,
    recycle_sender: SyncSender<Vec<i16>>,
    counters: &Arc<IoCounters>,
) -> Result<Stream, AudioError> {
    macro_rules! build {
        ($sample:ty) => {
            build_typed_output::<$sample>(device, config, receiver, recycle_sender, counters)
        };
    }
    match sample_format {
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I24 => build!(I24),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U24 => build!(U24),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        _ => Err(AudioError::new(
            AudioErrorKind::UnsupportedFormat,
            "speaker sample format is unsupported",
        )),
    }
}

fn build_typed_output<T>(
    device: &Device,
    config: &StreamConfig,
    receiver: Receiver<Vec<i16>>,
    recycle_sender: SyncSender<Vec<i16>>,
    counters: &Arc<IoCounters>,
) -> Result<Stream, AudioError>
where
    T: Sample + SizedSample + FromSample<i16>,
{
    let callback_counters = counters.clone();
    let error_counters = counters.clone();
    let mut playback = PlaybackBuffer::new(receiver, recycle_sender, callback_counters);
    device
        .build_output_stream(
            *config,
            move |output: &mut [T], _| playback.fill(output),
            move |_| {
                error_counters.stream_errors.fetch_add(1, Ordering::Relaxed);
            },
            Some(Duration::from_secs(2)),
        )
        .map_err(|_| device_error())
}

struct PlaybackBuffer {
    receiver: Receiver<Vec<i16>>,
    recycle_sender: SyncSender<Vec<i16>>,
    current: Vec<i16>,
    cursor: usize,
    disconnected: bool,
    counters: Arc<IoCounters>,
}

impl PlaybackBuffer {
    fn new(
        receiver: Receiver<Vec<i16>>,
        recycle_sender: SyncSender<Vec<i16>>,
        counters: Arc<IoCounters>,
    ) -> Self {
        Self {
            receiver,
            recycle_sender,
            current: Vec::new(),
            cursor: 0,
            disconnected: false,
            counters,
        }
    }

    fn fill<T: Sample + FromSample<i16>>(&mut self, output: &mut [T]) {
        let output_len = output.len();
        let mut underrun = false;
        let mut consumed = 0_u64;
        for sample in output {
            if self.cursor == self.current.len() && !self.disconnected {
                if !self.current.is_empty() {
                    self.current.clear();
                    let completed = std::mem::take(&mut self.current);
                    let _ = self.recycle_sender.try_send(completed);
                }
                match self.receiver.try_recv() {
                    Ok(chunk) => {
                        self.current = chunk;
                        self.cursor = 0;
                    }
                    Err(TryRecvError::Empty) => underrun = true,
                    Err(TryRecvError::Disconnected) => self.disconnected = true,
                }
            }
            let value = self.current.get(self.cursor).copied().unwrap_or(0);
            if self.cursor < self.current.len() {
                self.cursor += 1;
                consumed = consumed.saturating_add(1);
            }
            *sample = T::from_sample(value);
        }
        self.counters.played_samples.fetch_add(
            u64::try_from(output_len).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        subtract_queued(&self.counters.queued_samples, consumed);
        if underrun {
            self.counters
                .underrun_callbacks
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn subtract_queued(counter: &AtomicU64, samples: u64) {
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
        Some(queued.saturating_sub(samples))
    });
}

const fn device_error() -> AudioError {
    AudioError::new(AudioErrorKind::Backend, "audio device is unavailable")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc::{TryRecvError, sync_channel};

    use super::{FrameAssembler, IoCounters, PlaybackBuffer};

    #[test]
    fn callback_assembler_never_blocks_when_queue_is_full() {
        let counters = Arc::new(IoCounters::default());
        let (sender, receiver) = sync_channel(1);
        let (recycle_sender, recycle_receiver) = sync_channel(1);
        recycle_sender
            .try_send(Vec::with_capacity(4))
            .expect("pool primes");
        let mut assembler = FrameAssembler::new(
            sender,
            recycle_sender,
            recycle_receiver,
            4,
            counters.clone(),
        );
        assembler.push([1, 2, 3, 4, 5, 6, 7, 8].into_iter());
        assert_eq!(receiver.try_recv(), Ok(vec![1, 2, 3, 4]));
        assert_eq!(receiver.try_recv(), Err(TryRecvError::Empty));
        let snapshot = counters.snapshot();
        assert_eq!(snapshot.captured_chunks, 1);
        assert_eq!(snapshot.dropped_chunks, 1);
    }

    #[test]
    fn playback_uses_silence_on_underrun() {
        let counters = Arc::new(IoCounters::default());
        counters
            .queued_samples
            .store(2, std::sync::atomic::Ordering::Release);
        let (sender, receiver) = sync_channel(1);
        let (recycle_sender, _recycle_receiver) = sync_channel(2);
        sender.send(vec![10, 20]).expect("fixture queues");
        let mut playback = PlaybackBuffer::new(receiver, recycle_sender, counters.clone());
        let mut output = [0_i16; 4];
        playback.fill(&mut output);
        assert_eq!(output, [10, 20, 0, 0]);
        assert_eq!(counters.snapshot().underrun_callbacks, 1);
        assert_eq!(counters.snapshot().queued_samples, 0);
    }
}
