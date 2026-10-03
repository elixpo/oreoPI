use oreo_core::CancellationToken;

use crate::{AudioError, AudioErrorKind, AudioLimits, AudioSource, StreamingTranscriber};

/// Streams one bounded capture into a transcriber without retaining raw audio.
///
/// # Errors
///
/// Returns a redacted error for cancellation, format changes, capacity,
/// capture failures, or transcriber failures.
pub fn transcribe_source(
    source: &mut dyn AudioSource,
    transcriber: &mut dyn StreamingTranscriber,
    limits: AudioLimits,
    cancellation: &CancellationToken,
) -> Result<String, AudioError> {
    let limits = limits.validate()?;
    let format = source.format().validate()?;
    let max_samples = limits.max_samples(format)?;
    let mut consumed_samples = 0_usize;
    transcriber.begin(format)?;
    let transcription = (|| {
        while let Some(chunk) = source.next_chunk(cancellation)? {
            if cancellation.is_cancelled() {
                return Err(AudioError::new(
                    AudioErrorKind::Cancelled,
                    "audio transcription was cancelled",
                ));
            }
            if chunk.format() != format {
                return Err(AudioError::new(
                    AudioErrorKind::UnsupportedFormat,
                    "audio source changed format",
                ));
            }
            consumed_samples = consumed_samples
                .checked_add(chunk.samples().len())
                .filter(|count| *count <= max_samples)
                .ok_or_else(|| {
                    AudioError::new(AudioErrorKind::Capacity, "audio capture exceeds its limit")
                })?;
            transcriber.push(&chunk, cancellation)?;
        }
        transcriber.finish(cancellation)
    })();
    let transcript = match transcription {
        Ok(transcript) => transcript,
        Err(error) => {
            transcriber.abort();
            return Err(error);
        }
    };
    if transcript.len() > limits.max_transcript_bytes {
        return Err(AudioError::new(
            AudioErrorKind::Capacity,
            "audio transcript exceeds its limit",
        ));
    }
    if transcript.trim().is_empty() {
        return Err(AudioError::new(
            AudioErrorKind::NoSpeech,
            "no speech was detected",
        ));
    }
    Ok(transcript)
}

/// Converts streamed text deltas into bounded sentence-like TTS units.
pub struct SpeechChunker {
    pending: String,
    max_buffer_bytes: usize,
}

impl SpeechChunker {
    /// Creates a chunker using the response-buffer bound in the audio profile.
    ///
    /// # Errors
    ///
    /// Returns an invalid-config error when the profile is invalid.
    pub fn new(limits: AudioLimits) -> Result<Self, AudioError> {
        let limits = limits.validate()?;
        Ok(Self {
            pending: String::new(),
            max_buffer_bytes: limits.max_response_buffer_bytes,
        })
    }

    /// Adds a model text delta and returns all newly speakable chunks.
    ///
    /// # Errors
    ///
    /// Rejects input that would exceed the bounded pending buffer.
    pub fn push(&mut self, delta: &str) -> Result<Vec<String>, AudioError> {
        if self.pending.len().saturating_add(delta.len()) > self.max_buffer_bytes {
            return Err(AudioError::new(
                AudioErrorKind::Capacity,
                "speech response buffer is full",
            ));
        }
        self.pending.push_str(delta);
        let mut chunks = Vec::new();
        while let Some(end) = speakable_end(&self.pending) {
            let remainder = self.pending.split_off(end);
            let chunk = std::mem::replace(&mut self.pending, remainder);
            let chunk = chunk.trim();
            if !chunk.is_empty() {
                chunks.push(chunk.to_owned());
            }
        }
        Ok(chunks)
    }

    /// Flushes the final non-empty response fragment.
    #[must_use]
    pub fn finish(&mut self) -> Option<String> {
        let chunk = std::mem::take(&mut self.pending);
        let chunk = chunk.trim();
        (!chunk.is_empty()).then(|| chunk.to_owned())
    }

    #[must_use]
    pub fn pending_bytes(&self) -> usize {
        self.pending.len()
    }
}

fn speakable_end(text: &str) -> Option<usize> {
    text.char_indices().find_map(|(index, character)| {
        matches!(character, '.' | '?' | '!' | ';' | '\n').then(|| index + character.len_utf8())
    })
}

#[cfg(test)]
mod tests {
    use oreo_core::CancellationToken;

    use crate::{
        AudioError, AudioErrorKind, AudioFormat, AudioLimits, AudioSource, PcmChunk,
        StreamingTranscriber, WavSource,
    };

    use super::{SpeechChunker, transcribe_source};

    struct CountingTranscriber {
        samples: usize,
        aborted: bool,
    }

    impl StreamingTranscriber for CountingTranscriber {
        fn begin(&mut self, _format: AudioFormat) -> Result<(), AudioError> {
            Ok(())
        }

        fn push(
            &mut self,
            chunk: &PcmChunk,
            _cancellation: &CancellationToken,
        ) -> Result<(), AudioError> {
            self.samples += chunk.samples().len();
            Ok(())
        }

        fn finish(&mut self, _cancellation: &CancellationToken) -> Result<String, AudioError> {
            Ok(format!("{} samples", self.samples))
        }

        fn abort(&mut self) {
            self.aborted = true;
        }
    }

    struct EndlessSource {
        format: AudioFormat,
    }

    impl AudioSource for EndlessSource {
        fn format(&self) -> AudioFormat {
            self.format
        }

        fn next_chunk(
            &mut self,
            _cancellation: &CancellationToken,
        ) -> Result<Option<PcmChunk>, AudioError> {
            Ok(Some(PcmChunk::new(self.format, vec![0; 800])?))
        }
    }

    fn pcm_wav(sample_count: usize) -> Vec<u8> {
        let data_bytes = sample_count * 2;
        let file_bytes = 36 + data_bytes;
        let mut wav = Vec::with_capacity(file_bytes + 8);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(
            &u32::try_from(file_bytes)
                .expect("fixture size")
                .to_le_bytes(),
        );
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&16_000_u32.to_le_bytes());
        wav.extend_from_slice(&32_000_u32.to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(
            &u32::try_from(data_bytes)
                .expect("fixture size")
                .to_le_bytes(),
        );
        wav.resize(wav.len() + data_bytes, 0);
        wav
    }

    #[test]
    fn wav_stream_reaches_transcriber_without_audio_retention() {
        let input = pcm_wav(640);
        let mut source = WavSource::read(input.as_slice(), AudioLimits::sbc()).expect("WAV opens");
        let mut transcriber = CountingTranscriber {
            samples: 0,
            aborted: false,
        };
        let transcript = transcribe_source(
            &mut source,
            &mut transcriber,
            AudioLimits::sbc(),
            &CancellationToken::new(),
        )
        .expect("fixture transcribes");
        assert_eq!(transcript, "640 samples");
    }

    #[test]
    fn generic_source_cannot_exceed_capture_bound() {
        let format = AudioFormat {
            sample_rate_hz: 8_000,
            channels: 1,
        };
        let mut source = EndlessSource { format };
        let mut transcriber = CountingTranscriber {
            samples: 0,
            aborted: false,
        };
        let limits = AudioLimits {
            max_capture_seconds: 1,
            ..AudioLimits::sbc()
        };
        let error = transcribe_source(
            &mut source,
            &mut transcriber,
            limits,
            &CancellationToken::new(),
        )
        .expect_err("unbounded source fails");
        assert_eq!(error.kind, AudioErrorKind::Capacity);
        assert!(transcriber.aborted);
    }

    #[test]
    fn streamed_response_emits_complete_speakable_chunks() {
        let mut chunker = SpeechChunker::new(AudioLimits::sbc()).expect("chunker starts");
        assert!(chunker.push("Hello from ").expect("delta fits").is_empty());
        assert_eq!(
            chunker.push("Oreo. Second sentence").expect("delta fits"),
            vec!["Hello from Oreo."]
        );
        assert_eq!(chunker.finish().as_deref(), Some("Second sentence"));
        assert_eq!(chunker.pending_bytes(), 0);
    }

    #[test]
    fn streamed_response_buffer_is_bounded() {
        let limits = AudioLimits {
            max_response_buffer_bytes: 64,
            ..AudioLimits::sbc()
        };
        let mut chunker = SpeechChunker::new(limits).expect("chunker starts");
        let error = chunker
            .push(&"x".repeat(65))
            .expect_err("oversized delta fails");
        assert_eq!(error.kind, AudioErrorKind::Capacity);
        assert_eq!(chunker.pending_bytes(), 0);
    }
}
