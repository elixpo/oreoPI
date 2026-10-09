use oreo_core::CancellationToken;

use crate::{
    AudioError, AudioErrorKind, AudioLimits, ConversationDirective, ConversationLanguage,
    EnergyVad, OpenWakeWordDetector, PcmChunk, STT_FORMAT, StreamingTranscriber, VadConfig,
    VadDecision, VoskTranscriber, WakeAudioWindow, WakeIntentClassifier,
};

const COOLDOWN_SAMPLES: usize = 8_000;
const COMMAND_START_TIMEOUT_SAMPLES: usize = 5 * 16_000;
const CONVERSATION_WINDOW_SAMPLES: usize = 30 * 16_000;

#[derive(Clone, Debug, PartialEq)]
pub enum WakePipelineEvent {
    WakeAccepted { score: f32 },
    CommandReady { transcript: String },
    CommandTimedOut,
    ConversationEnded,
}

enum RuntimeState {
    Listening,
    Validating {
        score: f32,
        vad: EnergyVad,
        samples: usize,
    },
    Capturing {
        vad: EnergyVad,
        samples: usize,
        speech_started: bool,
    },
    Engaged {
        vad: EnergyVad,
        samples_left: usize,
    },
    EngagedCapturing {
        vad: EnergyVad,
        samples: usize,
        session_samples_left: usize,
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
    conversation: ConversationLanguage,
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
            conversation: ConversationLanguage::embedded()?,
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
        self.validate_chunk(chunk, cancellation)?;
        match &mut self.state {
            RuntimeState::Listening => {
                self.wake_window.push(chunk, cancellation)?;
                if let Some(candidate) = self.detector.push(chunk, cancellation)? {
                    let mut vad = EnergyVad::new(VadConfig::sbc())?;
                    let samples = self.begin_candidate(&mut vad, cancellation)?;
                    self.detector.reset(cancellation)?;
                    self.state = RuntimeState::Validating {
                        score: candidate.score,
                        vad,
                        samples,
                    };
                }
                Ok(None)
            }
            RuntimeState::Validating {
                score,
                vad,
                samples,
            } => {
                *samples = samples.saturating_add(chunk.samples().len());
                self.transcriber.push(chunk, cancellation)?;
                if vad.observe(chunk) != VadDecision::Endpoint
                    && *samples < self.max_command_samples
                {
                    return Ok(None);
                }
                let candidate_score = *score;
                self.finish_candidate(candidate_score, cancellation)
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
            RuntimeState::Engaged { vad, samples_left } => {
                self.wake_window.push(chunk, cancellation)?;
                *samples_left = samples_left.saturating_sub(chunk.samples().len());
                if vad.observe(chunk) == VadDecision::SpeechStarted {
                    let session_samples_left = *samples_left;
                    self.begin_engaged_capture(session_samples_left, cancellation)?;
                } else if *samples_left == 0 {
                    self.wake_window.clear();
                    self.detector.reset(cancellation)?;
                    self.state = RuntimeState::Listening;
                    return Ok(Some(WakePipelineEvent::ConversationEnded));
                }
                Ok(None)
            }
            RuntimeState::EngagedCapturing {
                vad,
                samples,
                session_samples_left,
            } => {
                *samples = samples.saturating_add(chunk.samples().len());
                *session_samples_left = session_samples_left.saturating_sub(chunk.samples().len());
                self.transcriber.push(chunk, cancellation)?;
                if vad.observe(chunk) == VadDecision::Endpoint
                    || *samples >= self.max_command_samples
                {
                    let remaining = *session_samples_left;
                    return self.finish_engaged_capture(remaining, cancellation);
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

    fn validate_chunk(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<(), AudioError> {
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
        Ok(())
    }

    pub fn shutdown(&mut self) {
        self.transcriber.abort();
        self.detector.shutdown();
        self.wake_window.clear();
    }

    fn begin_candidate(
        &mut self,
        vad: &mut EnergyVad,
        cancellation: &CancellationToken,
    ) -> Result<usize, AudioError> {
        let samples = self.wake_window.snapshot();
        self.transcriber.begin(STT_FORMAT)?;
        let transcription = (|| -> Result<(), AudioError> {
            for samples in samples.chunks(1_600) {
                let chunk = PcmChunk::new(STT_FORMAT, samples.to_vec())?;
                self.transcriber.push(&chunk, cancellation)?;
                let _ = vad.observe(&chunk);
            }
            Ok(())
        })();
        if transcription.is_err() {
            self.transcriber.abort();
        }
        transcription.map(|()| samples.len())
    }

    fn finish_candidate(
        &mut self,
        score: f32,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        let transcript = self.transcriber.finish(cancellation)?;
        let decision = (!transcript.trim().is_empty())
            .then(|| self.classifier.classify(&transcript))
            .transpose()?;
        self.wake_window.clear();
        if decision.is_some_and(|decision| decision.addressed)
            && self.classifier.is_identity_only(&transcript)
        {
            self.transcriber.begin(STT_FORMAT)?;
            self.state = RuntimeState::Capturing {
                vad: EnergyVad::new(VadConfig::sbc())?,
                samples: 0,
                speech_started: false,
            };
            Ok(Some(WakePipelineEvent::WakeAccepted { score }))
        } else if decision.is_some_and(|decision| decision.addressed) {
            self.complete_transcript(transcript, score, cancellation)
        } else {
            self.state = RuntimeState::Cooldown {
                samples_left: COOLDOWN_SAMPLES,
            };
            Ok(None)
        }
    }

    fn begin_engaged_capture(
        &mut self,
        session_samples_left: usize,
        cancellation: &CancellationToken,
    ) -> Result<(), AudioError> {
        let samples = self.wake_window.snapshot();
        self.transcriber.begin(STT_FORMAT)?;
        let mut capture_vad = EnergyVad::new(VadConfig::sbc())?;
        let transcription = (|| -> Result<(), AudioError> {
            for samples in samples.chunks(1_600) {
                let buffered = PcmChunk::new(STT_FORMAT, samples.to_vec())?;
                self.transcriber.push(&buffered, cancellation)?;
                let _ = capture_vad.observe(&buffered);
            }
            Ok(())
        })();
        if transcription.is_err() {
            self.transcriber.abort();
        }
        transcription?;
        self.wake_window.clear();
        self.state = RuntimeState::EngagedCapturing {
            vad: capture_vad,
            samples: samples.len(),
            session_samples_left,
        };
        Ok(())
    }

    fn finish_engaged_capture(
        &mut self,
        session_samples_left: usize,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        let transcript = self.transcriber.finish(cancellation)?;
        if !transcript.trim().is_empty() {
            return self.complete_transcript(transcript, 0.0, cancellation);
        }
        self.wake_window.clear();
        if session_samples_left == 0 {
            self.detector.reset(cancellation)?;
            self.state = RuntimeState::Listening;
            Ok(Some(WakePipelineEvent::ConversationEnded))
        } else {
            self.state = RuntimeState::Engaged {
                vad: EnergyVad::new(VadConfig::sbc())?,
                samples_left: session_samples_left,
            };
            Ok(None)
        }
    }

    fn finish_command(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        let transcript = self.transcriber.finish(cancellation)?;
        if transcript.trim().is_empty() {
            self.state = RuntimeState::Cooldown {
                samples_left: COOLDOWN_SAMPLES,
            };
            Ok(Some(WakePipelineEvent::CommandTimedOut))
        } else if transcript.len() > self.limits.max_transcript_bytes {
            Err(AudioError::new(
                AudioErrorKind::Capacity,
                "voice command transcript exceeds its limit",
            ))
        } else {
            self.complete_transcript(transcript, 0.0, cancellation)
        }
    }

    fn complete_transcript(
        &mut self,
        transcript: String,
        score: f32,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        if transcript.trim().is_empty() {
            self.state = RuntimeState::Listening;
            return Ok(Some(WakePipelineEvent::CommandTimedOut));
        }
        if transcript.len() > self.limits.max_transcript_bytes {
            return Err(AudioError::new(
                AudioErrorKind::Capacity,
                "voice command transcript exceeds its limit",
            ));
        }
        match self.conversation.classify(&transcript)? {
            ConversationDirective::Sleep => {
                self.wake_window.clear();
                self.detector.reset(cancellation)?;
                self.state = RuntimeState::Listening;
                Ok(Some(WakePipelineEvent::ConversationEnded))
            }
            ConversationDirective::AddressOnly => {
                self.transcriber.begin(STT_FORMAT)?;
                self.state = RuntimeState::Capturing {
                    vad: EnergyVad::new(VadConfig::sbc())?,
                    samples: 0,
                    speech_started: false,
                };
                Ok(Some(WakePipelineEvent::WakeAccepted { score }))
            }
            ConversationDirective::Command => {
                self.wake_window.clear();
                self.state = RuntimeState::Engaged {
                    vad: EnergyVad::new(VadConfig::sbc())?,
                    samples_left: CONVERSATION_WINDOW_SAMPLES,
                };
                Ok(Some(WakePipelineEvent::CommandReady { transcript }))
            }
        }
    }
}

impl Drop for WakeCommandPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}
