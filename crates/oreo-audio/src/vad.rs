use crate::{AudioError, AudioErrorKind, PcmChunk};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VadConfig {
    pub speech_threshold: i16,
    pub speech_start_frames: u16,
    pub silence_end_frames: u16,
}

impl VadConfig {
    #[must_use]
    pub const fn sbc() -> Self {
        Self {
            speech_threshold: 500,
            speech_start_frames: 2,
            silence_end_frames: 15,
        }
    }

    /// Validates deterministic endpoint bounds.
    ///
    /// # Errors
    ///
    /// Rejects zero/negative thresholds or frame counts outside a bounded
    /// two-second window at 10 ms per frame.
    pub fn validate(self) -> Result<Self, AudioError> {
        if self.speech_threshold <= 0
            || !(1..=20).contains(&self.speech_start_frames)
            || !(1..=200).contains(&self.silence_end_frames)
        {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "voice activity settings are invalid",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VadDecision {
    Silence,
    SpeechStarted,
    Speech,
    Endpoint,
}

/// Small deterministic energy detector used for fixtures and as a fallback.
pub struct EnergyVad {
    config: VadConfig,
    consecutive_speech: u16,
    consecutive_silence: u16,
    in_speech: bool,
}

impl EnergyVad {
    /// Creates a reset detector.
    ///
    /// # Errors
    ///
    /// Returns invalid-config when endpoint bounds are unusable.
    pub fn new(config: VadConfig) -> Result<Self, AudioError> {
        Ok(Self {
            config: config.validate()?,
            consecutive_speech: 0,
            consecutive_silence: 0,
            in_speech: false,
        })
    }

    #[must_use]
    pub const fn endpoint_frames(&self) -> u16 {
        self.config.silence_end_frames
    }

    #[must_use]
    pub const fn endpoint_ms(&self, frame_ms: u16) -> u32 {
        (self.config.silence_end_frames as u32).saturating_mul(frame_ms as u32)
    }

    #[must_use]
    pub fn observe(&mut self, chunk: &PcmChunk) -> VadDecision {
        let active = mean_absolute_amplitude(chunk.samples())
            >= u64::from(self.config.speech_threshold.unsigned_abs());
        if active {
            self.consecutive_silence = 0;
            self.consecutive_speech = self.consecutive_speech.saturating_add(1);
            if !self.in_speech && self.consecutive_speech >= self.config.speech_start_frames {
                self.in_speech = true;
                return VadDecision::SpeechStarted;
            }
            return if self.in_speech {
                VadDecision::Speech
            } else {
                VadDecision::Silence
            };
        }

        self.consecutive_speech = 0;
        if !self.in_speech {
            return VadDecision::Silence;
        }
        self.consecutive_silence = self.consecutive_silence.saturating_add(1);
        if self.consecutive_silence >= self.config.silence_end_frames {
            self.reset();
            VadDecision::Endpoint
        } else {
            VadDecision::Speech
        }
    }

    pub fn reset(&mut self) {
        self.consecutive_speech = 0;
        self.consecutive_silence = 0;
        self.in_speech = false;
    }
}

fn mean_absolute_amplitude(samples: &[i16]) -> u64 {
    let total = samples
        .iter()
        .map(|sample| u64::from(sample.unsigned_abs()))
        .sum::<u64>();
    total / u64::try_from(samples.len()).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use crate::{AudioFormat, AudioLimits, PcmChunk};

    use super::{EnergyVad, VadConfig, VadDecision};

    fn chunk(amplitude: i16) -> PcmChunk {
        PcmChunk::new(
            AudioFormat {
                sample_rate_hz: 16_000,
                channels: 1,
            },
            vec![amplitude; 320],
        )
        .expect("chunk is valid")
    }

    #[test]
    fn endpoint_occurs_after_three_hundred_milliseconds() {
        let mut vad = EnergyVad::new(VadConfig::sbc()).expect("VAD starts");
        assert_eq!(vad.observe(&chunk(800)), VadDecision::Silence);
        assert_eq!(vad.observe(&chunk(800)), VadDecision::SpeechStarted);
        for _ in 0..14 {
            assert_eq!(vad.observe(&chunk(0)), VadDecision::Speech);
        }
        assert_eq!(vad.observe(&chunk(0)), VadDecision::Endpoint);
        assert_eq!(vad.endpoint_frames(), 15);
        assert_eq!(vad.endpoint_ms(AudioLimits::sbc().frame_ms), 300);
        assert!(vad.endpoint_ms(AudioLimits::sbc().frame_ms) < 500);
    }

    #[test]
    fn isolated_noise_does_not_start_speech() {
        let mut vad = EnergyVad::new(VadConfig::sbc()).expect("VAD starts");
        assert_eq!(vad.observe(&chunk(2_000)), VadDecision::Silence);
        assert_eq!(vad.observe(&chunk(0)), VadDecision::Silence);
    }
}
