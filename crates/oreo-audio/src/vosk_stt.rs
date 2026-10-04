use std::fmt;
use std::path::Path;

use oreo_core::CancellationToken;
use vosk::{CompleteResult, DecodingState, LogLevel, Model, Recognizer};

use crate::{
    AudioError, AudioErrorKind, AudioFormat, AudioLimits, PcmChunk, STT_FORMAT,
    StreamingTranscriber,
};

/// Bounded Vosk adapter for push-to-talk utterances.
///
/// The model remains loaded between turns, while every call to `begin` creates
/// fresh recognizer state. Raw PCM is passed directly to Vosk and is never
/// retained by this adapter.
pub struct VoskTranscriber {
    // Drop recognizer before model: Vosk recognizers refer to shared model data.
    recognizer: Option<Recognizer>,
    model: Model,
    transcript: String,
    confidence: ConfidenceAccumulator,
    max_transcript_bytes: usize,
}

/// Aggregate word confidence from one completed Vosk utterance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoskConfidence {
    pub mean: f32,
    pub minimum: f32,
    pub words: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct ConfidenceAccumulator {
    sum: f32,
    count: f32,
    words: u32,
    minimum: Option<f32>,
}

impl ConfidenceAccumulator {
    fn record(&mut self, confidence: f32) {
        self.sum += confidence;
        self.count += 1.0;
        self.words = self.words.saturating_add(1);
        self.minimum = Some(
            self.minimum
                .map_or(confidence, |value| value.min(confidence)),
        );
    }

    fn merge(&mut self, other: Self) {
        self.sum += other.sum;
        self.count += other.count;
        self.words = self.words.saturating_add(other.words);
        if let Some(minimum) = other.minimum {
            self.minimum = Some(self.minimum.map_or(minimum, |value| value.min(minimum)));
        }
    }

    fn complete(self) -> Option<VoskConfidence> {
        Some(VoskConfidence {
            mean: self.sum / self.count,
            minimum: self.minimum?,
            words: self.words,
        })
    }
}

struct DecodedSegment {
    text: String,
    confidence: ConfidenceAccumulator,
}

impl fmt::Debug for VoskTranscriber {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VoskTranscriber")
            .field("active", &self.recognizer.is_some())
            .field("transcript_bytes", &self.transcript.len())
            .field("confidence_words", &self.confidence.words)
            .field("max_transcript_bytes", &self.max_transcript_bytes)
            .finish_non_exhaustive()
    }
}

impl VoskTranscriber {
    /// Loads a Vosk model once for reuse across push-to-talk utterances.
    ///
    /// # Errors
    ///
    /// Returns a redacted error for invalid limits, a non-UTF-8 model path, or
    /// a model directory Vosk cannot load.
    pub fn load(model_path: impl AsRef<Path>, limits: AudioLimits) -> Result<Self, AudioError> {
        let limits = limits.validate()?;
        let model_path = model_path.as_ref().to_str().ok_or_else(|| {
            AudioError::new(AudioErrorKind::InvalidConfig, "Vosk model path is invalid")
        })?;
        vosk::set_log_level(LogLevel::Error);
        let model = Model::new(model_path.to_owned()).ok_or_else(|| {
            AudioError::new(AudioErrorKind::Backend, "Vosk model could not be loaded")
        })?;
        Ok(Self {
            recognizer: None,
            model,
            transcript: String::with_capacity(limits.max_transcript_bytes),
            confidence: ConfidenceAccumulator::default(),
            max_transcript_bytes: limits.max_transcript_bytes,
        })
    }

    #[must_use]
    pub fn is_active(&self) -> bool {
        self.recognizer.is_some()
    }

    /// Returns aggregate confidence after a successful completed utterance.
    #[must_use]
    pub fn confidence(&self) -> Option<VoskConfidence> {
        if self.recognizer.is_some() {
            None
        } else {
            self.confidence.complete()
        }
    }

    fn reset_utterance(&mut self) {
        self.recognizer = None;
        self.transcript.clear();
        self.confidence = ConfidenceAccumulator::default();
    }

    fn append_segment(&mut self, segment: &DecodedSegment) -> Result<(), AudioError> {
        append_bounded(
            &mut self.transcript,
            &segment.text,
            self.max_transcript_bytes,
        )?;
        self.confidence.merge(segment.confidence);
        Ok(())
    }
}

impl StreamingTranscriber for VoskTranscriber {
    fn begin(&mut self, format: AudioFormat) -> Result<(), AudioError> {
        if self.recognizer.is_some() {
            return Err(AudioError::new(
                AudioErrorKind::InvalidTransition,
                "Vosk transcription is already active",
            ));
        }
        if format != STT_FORMAT {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "Vosk requires 16 kHz mono PCM",
            ));
        }
        self.transcript.clear();
        self.confidence = ConfidenceAccumulator::default();
        let mut recognizer = Recognizer::new(&self.model, 16_000.0_f32).ok_or_else(|| {
            AudioError::new(
                AudioErrorKind::Backend,
                "Vosk recognizer could not be created",
            )
        })?;
        recognizer.set_words(true);
        self.recognizer = Some(recognizer);
        Ok(())
    }

    fn push(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<(), AudioError> {
        if cancellation.is_cancelled() {
            self.reset_utterance();
            return Err(cancelled());
        }
        if chunk.format() != STT_FORMAT {
            self.reset_utterance();
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "Vosk requires 16 kHz mono PCM",
            ));
        }

        let segment = (|| -> Result<Option<DecodedSegment>, AudioError> {
            let recognizer = self.recognizer.as_mut().ok_or_else(|| {
                AudioError::new(
                    AudioErrorKind::InvalidTransition,
                    "Vosk transcription is not active",
                )
            })?;
            match recognizer.accept_waveform(chunk.samples()).map_err(|_| {
                AudioError::new(AudioErrorKind::Backend, "Vosk transcription failed")
            })? {
                DecodingState::Running => Ok(None),
                DecodingState::Finalized => Ok(Some(decode_segment(recognizer.result())?)),
                DecodingState::Failed => Err(AudioError::new(
                    AudioErrorKind::Backend,
                    "Vosk transcription failed",
                )),
            }
        })();
        let segment = match segment {
            Ok(segment) => segment,
            Err(error) => {
                self.reset_utterance();
                return Err(error);
            }
        };

        if cancellation.is_cancelled() {
            self.reset_utterance();
            return Err(cancelled());
        }
        if let Some(segment) = segment
            && let Err(error) = self.append_segment(&segment)
        {
            self.reset_utterance();
            return Err(error);
        }
        Ok(())
    }

    fn abort(&mut self) {
        self.reset_utterance();
    }

    fn finish(&mut self, cancellation: &CancellationToken) -> Result<String, AudioError> {
        if cancellation.is_cancelled() {
            self.reset_utterance();
            return Err(cancelled());
        }
        let mut recognizer = self.recognizer.take().ok_or_else(|| {
            AudioError::new(
                AudioErrorKind::InvalidTransition,
                "Vosk transcription is not active",
            )
        })?;
        let final_segment = match decode_segment(recognizer.final_result()) {
            Ok(segment) => segment,
            Err(error) => {
                self.reset_utterance();
                return Err(error);
            }
        };
        drop(recognizer);
        if cancellation.is_cancelled() {
            self.reset_utterance();
            return Err(cancelled());
        }
        if let Err(error) = self.append_segment(&final_segment) {
            self.reset_utterance();
            return Err(error);
        }
        Ok(std::mem::take(&mut self.transcript))
    }
}

fn decode_segment(result: CompleteResult<'_>) -> Result<DecodedSegment, AudioError> {
    let result = result.single().ok_or_else(|| {
        AudioError::new(AudioErrorKind::Backend, "Vosk returned an invalid result")
    })?;
    let mut confidence = ConfidenceAccumulator::default();
    for word in result.result {
        confidence.record(word.conf);
    }
    Ok(DecodedSegment {
        text: result.text.trim().to_owned(),
        confidence,
    })
}

fn append_bounded(
    transcript: &mut String,
    segment: &str,
    max_bytes: usize,
) -> Result<(), AudioError> {
    let segment = segment.trim();
    if segment.is_empty() {
        return Ok(());
    }
    let separator_bytes = usize::from(!transcript.is_empty());
    if transcript
        .len()
        .checked_add(separator_bytes)
        .and_then(|bytes| bytes.checked_add(segment.len()))
        .is_none_or(|bytes| bytes > max_bytes)
    {
        return Err(AudioError::new(
            AudioErrorKind::Capacity,
            "Vosk transcript exceeds its limit",
        ));
    }
    if separator_bytes != 0 {
        transcript.push(' ');
    }
    transcript.push_str(segment);
    Ok(())
}

fn cancelled() -> AudioError {
    AudioError::new(
        AudioErrorKind::Cancelled,
        "Vosk transcription was cancelled",
    )
}

#[cfg(test)]
mod tests {
    use super::{ConfidenceAccumulator, append_bounded};
    use crate::AudioErrorKind;

    #[test]
    fn bounded_segments_join_without_retaining_empty_results() {
        let mut transcript = String::new();
        append_bounded(&mut transcript, " first segment ", 32).expect("segment fits");
        append_bounded(&mut transcript, "", 32).expect("empty segment is ignored");
        append_bounded(&mut transcript, "second", 32).expect("segment fits");
        assert_eq!(transcript, "first segment second");
    }

    #[test]
    fn bounded_segments_reject_growth_without_modifying_transcript() {
        let mut transcript = "keep".to_owned();
        let error = append_bounded(&mut transcript, "oversized", 8).expect_err("must fail");
        assert_eq!(error.kind, AudioErrorKind::Capacity);
        assert_eq!(transcript, "keep");
    }

    #[test]
    fn confidence_accumulator_tracks_mean_minimum_and_count() {
        let mut confidence = ConfidenceAccumulator::default();
        confidence.record(0.9);
        confidence.record(0.7);
        let complete = confidence.complete().expect("confidence exists");
        assert!((complete.mean - 0.8).abs() < f32::EPSILON);
        assert!((complete.minimum - 0.7).abs() < f32::EPSILON);
        assert_eq!(complete.words, 2);
    }
}
