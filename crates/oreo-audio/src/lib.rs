//! Bounded, backend-neutral contracts for Oreo's local audio pipeline.
//!
//! Raw PCM exists only in short-lived chunks. This crate has no persistence,
//! network, credential, or model-loading API.

mod converter;
mod cpal_io;
mod metrics;
mod pipeline;
mod pronunciation;
mod stream;
mod vad;
mod wav;

use std::error::Error;
use std::fmt;

use oreo_core::CancellationToken;

pub use converter::{ConvertingSource, PcmConverter};
pub use cpal_io::{
    AudioDeviceSummary, AudioIoSnapshot, CaptureControl, CpalInputSource, CpalOutput,
    default_audio_devices,
};
pub use metrics::{LatencyMetric, LatencyReport, LatencyTargets, LatencyWindow};
pub use pipeline::{PipelinePhase, PushToTalkState};
pub use pronunciation::normalize_for_speech;
pub use stream::{SpeechChunker, transcribe_source};
pub use vad::{EnergyVad, VadConfig, VadDecision};
pub use wav::WavSource;

pub const MAX_CHANNELS: u16 = 2;
pub const MIN_SAMPLE_RATE_HZ: u32 = 8_000;
pub const MAX_SAMPLE_RATE_HZ: u32 = 48_000;
pub const STT_FORMAT: AudioFormat = AudioFormat {
    sample_rate_hz: 16_000,
    channels: 1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioFormat {
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl AudioFormat {
    /// Validates a PCM format supported by the v0.1 voice path.
    ///
    /// # Errors
    ///
    /// Rejects sample rates outside 8-48 kHz and channel counts outside 1-2.
    pub fn validate(self) -> Result<Self, AudioError> {
        if !(MIN_SAMPLE_RATE_HZ..=MAX_SAMPLE_RATE_HZ).contains(&self.sample_rate_hz)
            || !(1..=MAX_CHANNELS).contains(&self.channels)
        {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "audio format is unsupported",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioLimits {
    pub frame_ms: u16,
    pub max_capture_seconds: u16,
    pub queue_capacity: usize,
    pub max_transcript_bytes: usize,
    pub max_response_buffer_bytes: usize,
}

impl AudioLimits {
    #[must_use]
    pub const fn sbc() -> Self {
        Self {
            frame_ms: 20,
            max_capture_seconds: 30,
            queue_capacity: 32,
            max_transcript_bytes: 4_096,
            max_response_buffer_bytes: 4_096,
        }
    }

    /// Validates fixed bounds before capture begins.
    ///
    /// # Errors
    ///
    /// Rejects zero or excessive frame, capture, and queue settings.
    pub fn validate(self) -> Result<Self, AudioError> {
        if !(10..=100).contains(&self.frame_ms)
            || !(1..=120).contains(&self.max_capture_seconds)
            || !(1..=256).contains(&self.queue_capacity)
            || !(64..=65_536).contains(&self.max_transcript_bytes)
            || !(64..=65_536).contains(&self.max_response_buffer_bytes)
        {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "audio limits are invalid",
            ));
        }
        Ok(self)
    }

    fn max_samples(self, format: AudioFormat) -> Result<usize, AudioError> {
        usize::try_from(format.sample_rate_hz)
            .ok()
            .and_then(|rate| rate.checked_mul(usize::from(format.channels)))
            .and_then(|per_second| per_second.checked_mul(usize::from(self.max_capture_seconds)))
            .ok_or_else(|| {
                AudioError::new(AudioErrorKind::InvalidConfig, "audio limit is too large")
            })
    }

    fn samples_per_chunk(self, format: AudioFormat) -> Result<usize, AudioError> {
        usize::try_from(format.sample_rate_hz)
            .ok()
            .and_then(|rate| rate.checked_mul(usize::from(self.frame_ms)))
            .and_then(|samples| samples.checked_div(1_000))
            .and_then(|frames| frames.checked_mul(usize::from(format.channels)))
            .filter(|samples| *samples > 0)
            .ok_or_else(|| AudioError::new(AudioErrorKind::InvalidConfig, "audio frame is invalid"))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct PcmChunk {
    format: AudioFormat,
    samples: Vec<i16>,
}

impl fmt::Debug for PcmChunk {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PcmChunk")
            .field("format", &self.format)
            .field("sample_count", &self.samples.len())
            .finish()
    }
}

impl PcmChunk {
    /// Creates one interleaved signed 16-bit PCM chunk.
    ///
    /// # Errors
    ///
    /// Rejects an invalid format, empty audio, or a chunk larger than 100 ms.
    pub fn new(format: AudioFormat, samples: Vec<i16>) -> Result<Self, AudioError> {
        let format = format.validate()?;
        let max_samples = usize::try_from(format.sample_rate_hz)
            .ok()
            .and_then(|rate| rate.checked_div(10))
            .and_then(|frames| frames.checked_mul(usize::from(format.channels)))
            .ok_or_else(|| {
                AudioError::new(AudioErrorKind::InvalidConfig, "audio frame is invalid")
            })?;
        if samples.is_empty()
            || samples.len() > max_samples
            || !samples.len().is_multiple_of(usize::from(format.channels))
        {
            return Err(AudioError::new(
                AudioErrorKind::InvalidFrame,
                "PCM chunk is invalid",
            ));
        }
        Ok(Self { format, samples })
    }

    #[must_use]
    pub const fn format(&self) -> AudioFormat {
        self.format
    }

    #[must_use]
    pub fn samples(&self) -> &[i16] {
        &self.samples
    }
}

pub trait AudioSource {
    /// Returns the source PCM format before the first chunk is read.
    fn format(&self) -> AudioFormat;

    /// Returns one bounded chunk, or `None` after push-to-talk release/end.
    ///
    /// # Errors
    ///
    /// Returns a redacted capture or cancellation failure.
    fn next_chunk(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Option<PcmChunk>, AudioError>;
}

pub trait StreamingTranscriber {
    /// Starts a fresh utterance in the supplied PCM format.
    ///
    /// # Errors
    ///
    /// Returns a redacted backend or format failure.
    fn begin(&mut self, format: AudioFormat) -> Result<(), AudioError>;

    /// Consumes one transient PCM chunk without retaining it after processing.
    ///
    /// # Errors
    ///
    /// Returns a redacted backend, capacity, or cancellation failure.
    fn push(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<(), AudioError>;

    /// Finalizes the utterance and returns bounded text.
    ///
    /// # Errors
    ///
    /// Returns a redacted backend or cancellation failure.
    fn finish(&mut self, cancellation: &CancellationToken) -> Result<String, AudioError>;
}

pub trait StreamingSynthesizer {
    /// Streams generated PCM without requiring the whole utterance in memory.
    ///
    /// # Errors
    ///
    /// Returns a redacted backend, output, or cancellation failure.
    fn synthesize(
        &mut self,
        text: &str,
        cancellation: &CancellationToken,
        emit: &mut dyn FnMut(PcmChunk) -> Result<(), AudioError>,
    ) -> Result<(), AudioError>;
}

pub trait AudioOutput {
    /// Begins cancellable playback for the supplied format.
    ///
    /// # Errors
    ///
    /// Returns a redacted device or format failure.
    fn begin(&mut self, format: AudioFormat) -> Result<(), AudioError>;

    /// Writes one transient chunk to the output device.
    ///
    /// # Errors
    ///
    /// Returns a redacted device, capacity, or cancellation failure.
    fn write(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<(), AudioError>;

    /// Stops playback and releases backend resources.
    fn stop(&mut self);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioErrorKind {
    InvalidConfig,
    UnsupportedFormat,
    InvalidFrame,
    InvalidTransition,
    Capacity,
    Cancelled,
    Input,
    Backend,
    NoSpeech,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioError {
    pub kind: AudioErrorKind,
    message: &'static str,
}

impl AudioError {
    pub(crate) const fn new(kind: AudioErrorKind, message: &'static str) -> Self {
        Self { kind, message }
    }
}

impl fmt::Display for AudioError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl Error for AudioError {}

#[cfg(test)]
mod tests {
    use super::{AudioErrorKind, AudioFormat, AudioLimits, PcmChunk};

    #[test]
    fn sbc_limits_are_valid_and_chunks_are_bounded() {
        let limits = AudioLimits::sbc().validate().expect("limits are valid");
        let format = AudioFormat {
            sample_rate_hz: 16_000,
            channels: 1,
        };
        assert_eq!(limits.samples_per_chunk(format), Ok(320));
        assert_eq!(limits.max_samples(format), Ok(480_000));
        assert!(PcmChunk::new(format, vec![0; 1_600]).is_ok());
        let error = PcmChunk::new(format, vec![0; 1_601]).expect_err("chunk is too large");
        assert_eq!(error.kind, AudioErrorKind::InvalidFrame);

        let chunk = PcmChunk::new(format, vec![12_345; 10]).expect("chunk is valid");
        let debug = format!("{chunk:?}");
        assert!(debug.contains("sample_count: 10"));
        assert!(!debug.contains("12345"));
    }
}
