use oreo_core::CancellationToken;

use crate::{
    AudioError, AudioErrorKind, AudioLimits, EnergyVad, OpenWakeWordDetector, PcmChunk, STT_FORMAT,
    StreamingTranscriber, VadConfig, VadDecision, VoskTranscriber, WakeAudioWindow,
    WakeIntentClassifier,
};

const POST_ROLL_SAMPLES: usize = 16_000;
const COOLDOWN_SAMPLES: usize = 16_000;
const COMMAND_START_TIMEOUT_SAMPLES: usize = 5 * 16_000;

#[derive(Clone, Debug, PartialEq)]
pub enum WakePipelineEvent {
    WakeAccepted { score: f32 },
    CommandReady { transcript: String },
    CommandTimedOut,
}

enum RuntimeState {
    Listening,
    Candidate {
        score: f32,
        post_roll_left: usize,
    },
    Capturing {
        vad: EnergyVad,
        samples: usize,
        speech_started: bool,
    },
    Cooldown {
        samples_left: usize,
    },
}

/// Owns the bounded wake-to-follow-up-command state machine.
///
/// Acoustic inference stays in the isolated openWakeWord worker. Candidate and
/// command PCM are transcribed by the reusable in-process Vosk model, while no
/// raw audio is persisted.
pub struct WakeCommandPipeline {
    detector: OpenWakeWordDetector,
    transcriber: VoskTranscriber,
    classifier: WakeIntentClassifier,
    wake_window: WakeAudioWindow,
    state: RuntimeState,
    limits: AudioLimits,
    max_command_samples: usize,
}

impl WakeCommandPipeline {
    /// Creates a reset listening pipeline from already-configured adapters.
    ///
    /// # Errors
    ///
    /// Rejects invalid audio bounds or an invalid embedded intent corpus.
    pub fn new(
        detector: OpenWakeWordDetector,
        transcriber: VoskTranscriber,
        limits: AudioLimits,
    ) -> Result<Self, AudioError> {
        let limits = limits.validate()?;
        let max_command_samples = usize::from(limits.max_capture_seconds)
            .checked_mul(16_000)
            .ok_or_else(|| {
                AudioError::new(
                    AudioErrorKind::InvalidConfig,
                    "voice capture limit is invalid",
                )
            })?;
        Ok(Self {
            detector,
            transcriber,
            classifier: WakeIntentClassifier::embedded()?,
            wake_window: WakeAudioWindow::new(3)?,
            state: RuntimeState::Listening,
            limits,
            max_command_samples,
        })
    }

    /// Warms the acoustic worker before microphone processing begins.
    ///
    /// # Errors
    ///
    /// Returns a redacted worker or cancellation failure.
    pub fn prewarm(&mut self, cancellation: &CancellationToken) -> Result<(), AudioError> {
        self.detector.prewarm(cancellation)
    }

    /// Processes one 16 kHz mono chunk and optionally emits a state transition.
    ///
    /// # Errors
    ///
    /// Fails closed on format, worker, Vosk, capacity, or cancellation errors.
    pub fn process(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        if chunk.format() != STT_FORMAT {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "voice runtime requires 16 kHz mono PCM",
            ));
        }
        if cancellation.is_cancelled() {
            self.transcriber.abort();
            self.wake_window.clear();
            return Err(AudioError::new(
                AudioErrorKind::Cancelled,
                "voice runtime was cancelled",
            ));
        }

        match &mut self.state {
            RuntimeState::Listening => {
                self.wake_window.push(chunk, cancellation)?;
                if let Some(candidate) = self.detector.push(chunk, cancellation)? {
                    self.state = RuntimeState::Candidate {
                        score: candidate.score,
                        post_roll_left: POST_ROLL_SAMPLES,
                    };
                }
                Ok(None)
            }
            RuntimeState::Candidate {
                score,
                post_roll_left,
            } => {
                self.wake_window.push(chunk, cancellation)?;
                *post_roll_left = post_roll_left.saturating_sub(chunk.samples().len());
                if *post_roll_left != 0 {
                    return Ok(None);
                }
                let candidate_score = *score;
                let transcript = self.transcribe_wake_window(cancellation)?;
                let accepted = if transcript.trim().is_empty() {
                    false
                } else {
                    self.classifier.classify(&transcript)?.addressed
                };
                self.detector.reset(cancellation)?;
                self.wake_window.clear();
                if accepted {
                    self.transcriber.begin(STT_FORMAT)?;
                    self.state = RuntimeState::Capturing {
                        vad: EnergyVad::new(VadConfig::sbc())?,
                        samples: 0,
                        speech_started: false,
                    };
                    Ok(Some(WakePipelineEvent::WakeAccepted {
                        score: candidate_score,
                    }))
                } else {
                    self.state = RuntimeState::Cooldown {
                        samples_left: COOLDOWN_SAMPLES,
                    };
                    Ok(None)
                }
            }
            RuntimeState::Capturing {
                vad,
                samples,
                speech_started,
            } => {
                *samples = samples.saturating_add(chunk.samples().len());
                self.transcriber.push(chunk, cancellation)?;
                match vad.observe(chunk) {
                    VadDecision::SpeechStarted => *speech_started = true,
                    VadDecision::Endpoint if *speech_started => {
                        return self.finish_command(cancellation);
                    }
                    VadDecision::Silence | VadDecision::Speech | VadDecision::Endpoint => {}
                }
                if (!*speech_started && *samples >= COMMAND_START_TIMEOUT_SAMPLES)
                    || *samples >= self.max_command_samples
                {
                    if *speech_started {
                        return self.finish_command(cancellation);
                    }
                    self.transcriber.abort();
                    self.state = RuntimeState::Cooldown {
                        samples_left: COOLDOWN_SAMPLES,
                    };
                    return Ok(Some(WakePipelineEvent::CommandTimedOut));
                }
                Ok(None)
            }
            RuntimeState::Cooldown { samples_left } => {
                *samples_left = samples_left.saturating_sub(chunk.samples().len());
                if *samples_left == 0 {
                    self.state = RuntimeState::Listening;
                }
                Ok(None)
            }
        }
    }

    pub fn shutdown(&mut self) {
        self.transcriber.abort();
        self.detector.shutdown();
        self.wake_window.clear();
    }

    fn transcribe_wake_window(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<String, AudioError> {
        let samples = self.wake_window.snapshot();
        self.transcriber.begin(STT_FORMAT)?;
        let transcription = (|| {
            for samples in samples.chunks(1_600) {
                let chunk = PcmChunk::new(STT_FORMAT, samples.to_vec())?;
                self.transcriber.push(&chunk, cancellation)?;
            }
            self.transcriber.finish(cancellation)
        })();
        if transcription.is_err() {
            self.transcriber.abort();
        }
        transcription
    }

    fn finish_command(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        let transcript = self.transcriber.finish(cancellation)?;
        self.state = RuntimeState::Cooldown {
            samples_left: COOLDOWN_SAMPLES,
        };
        if transcript.trim().is_empty() {
            Ok(Some(WakePipelineEvent::CommandTimedOut))
        } else if transcript.len() > self.limits.max_transcript_bytes {
            Err(AudioError::new(
                AudioErrorKind::Capacity,
                "voice command transcript exceeds its limit",
            ))
        } else {
            Ok(Some(WakePipelineEvent::CommandReady { transcript }))
        }
    }
}

impl Drop for WakeCommandPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}
