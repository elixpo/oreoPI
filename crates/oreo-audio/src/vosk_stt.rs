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
    max_transcript_bytes: usize,
}

impl fmt::Debug for VoskTranscriber {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VoskTranscriber")
            .field("active", &self.recognizer.is_some())
            .field("transcript_bytes", &self.transcript.len())
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
            max_transcript_bytes: limits.max_transcript_bytes,
        })
    }

    #[must_use]
    pub fn is_active(&self) -> bool {
        self.recognizer.is_some()
    }

    fn reset_utterance(&mut self) {
        self.recognizer = None;
        self.transcript.clear();
    }

    fn append_segment(&mut self, segment: &str) -> Result<(), AudioError> {
        append_bounded(&mut self.transcript, segment, self.max_transcript_bytes)
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
        self.recognizer = Recognizer::new(&self.model, 16_000.0_f32);
        if self.recognizer.is_none() {
            return Err(AudioError::new(
                AudioErrorKind::Backend,
                "Vosk recognizer could not be created",
            ));
        }
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

        let segment = (|| -> Result<Option<String>, AudioError> {
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
                DecodingState::Finalized => Ok(Some(single_text(recognizer.result())?)),
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
        let final_segment = match single_text(recognizer.final_result()) {
            Ok(segment) => segment,
            Err(error) => {
                self.transcript.clear();
                return Err(error);
            }
        };
        drop(recognizer);
        if cancellation.is_cancelled() {
            self.transcript.clear();
            return Err(cancelled());
        }
        if let Err(error) = self.append_segment(&final_segment) {
            self.transcript.clear();
            return Err(error);
        }
        Ok(std::mem::take(&mut self.transcript))
    }
}

fn single_text(result: CompleteResult<'_>) -> Result<String, AudioError> {
    result
        .single()
        .map(|result| result.text.trim().to_owned())
        .ok_or_else(|| AudioError::new(AudioErrorKind::Backend, "Vosk returned an invalid result"))
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
    use super::append_bounded;
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
}
