use oreo_core::CancellationToken;

use crate::{
    AudioError, AudioErrorKind, AudioLimits, ConversationDirective, ConversationLanguage,
    EnergyVad, OpenWakeWordDetector, PcmChunk, STT_FORMAT, StreamingTranscriber, VadConfig,
    VadDecision, VoskTranscriber, WakeAudioWindow, WakeIntentClassifier, WakeIntentDisposition,
    WarmTurnClassifier,
};

const COOLDOWN_SAMPLES: usize = 8_000;
const COMMAND_START_TIMEOUT_SAMPLES: usize = 5 * 16_000;
const CONVERSATION_WINDOW_SAMPLES: usize = 30 * 16_000;

#[derive(Clone, Debug, PartialEq)]
pub enum WakePipelineEvent {
    WakeAccepted {
        score: f32,
    },
    CommandReady {
        transcript: String,
    },
    CommandTimedOut,
    ConversationEnded,
    ClarificationNeeded,
    /// A non-trivial live utterance was heard while Oreo was speaking. The
    /// daemon compares it with the private playback reference before ducking
    /// output; the partial text is never logged or persisted.
    PossibleBargeIn {
        transcript: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConversationAudioPhase {
    Listening,
    Thinking,
    Speaking,
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
        last_live_transcript: String,
    },
    Cooldown {
        samples_left: usize,
    },
}

enum EngagedCaptureProgress {
    Continue,
    BargeIn(String),
    Finished(usize),
}

enum CommandCaptureProgress {
    Continue,
    Finish,
    TimedOut,
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
    warm_classifier: WarmTurnClassifier,
    conversation: ConversationLanguage,
    wake_window: WakeAudioWindow,
    state: RuntimeState,
    limits: AudioLimits,
    max_command_samples: usize,
    conversation_phase: ConversationAudioPhase,
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
            warm_classifier: WarmTurnClassifier::embedded()?,
            conversation: ConversationLanguage::embedded()?,
            wake_window: WakeAudioWindow::new(3)?,
            state: RuntimeState::Listening,
            limits,
            max_command_samples,
            conversation_phase: ConversationAudioPhase::Listening,
        })
    }

    /// Holds the warm conversation window open while the agent is thinking or
    /// speaking. The microphone continues processing possible steering turns.
    pub fn set_conversation_phase(&mut self, phase: ConversationAudioPhase) {
        self.conversation_phase = phase;
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
        let conversation_busy = self.conversation_phase != ConversationAudioPhase::Listening;
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
            } => match advance_command_capture(
                &mut self.transcriber,
                self.max_command_samples,
                chunk,
                cancellation,
                vad,
                samples,
                speech_started,
            )? {
                CommandCaptureProgress::Continue => Ok(None),
                CommandCaptureProgress::Finish => self.finish_command(cancellation),
                CommandCaptureProgress::TimedOut => {
                    self.transcriber.abort();
                    self.state = RuntimeState::Cooldown {
                        samples_left: COOLDOWN_SAMPLES,
                    };
                    Ok(Some(WakePipelineEvent::CommandTimedOut))
                }
            },
            RuntimeState::Engaged { vad, samples_left } => {
                self.wake_window.push(chunk, cancellation)?;
                consume_session_time(samples_left, chunk.samples().len(), conversation_busy);
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
                last_live_transcript,
            } => match advance_engaged_capture(
                &mut self.transcriber,
                self.conversation_phase,
                self.max_command_samples,
                chunk,
                cancellation,
                vad,
                samples,
                session_samples_left,
                last_live_transcript,
            )? {
                EngagedCaptureProgress::Continue => Ok(None),
                EngagedCaptureProgress::BargeIn(transcript) => Ok(Some(barge_in(transcript))),
                EngagedCaptureProgress::Finished(remaining) => {
                    self.finish_engaged_capture(remaining, cancellation)
                }
            },
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
        let disposition = decision.map(|decision| decision.disposition);
        if disposition == Some(WakeIntentDisposition::Addressed)
            && self.classifier.is_identity_only(&transcript)
        {
            self.transcriber.begin(STT_FORMAT)?;
            self.state = RuntimeState::Capturing {
                vad: EnergyVad::new(VadConfig::conversation())?,
                samples: 0,
                speech_started: false,
            };
            Ok(Some(WakePipelineEvent::WakeAccepted { score }))
        } else if disposition == Some(WakeIntentDisposition::Addressed) {
            self.complete_transcript(transcript, score, cancellation)
        } else if disposition == Some(WakeIntentDisposition::Clarify) {
            self.state = RuntimeState::Cooldown {
                samples_left: COOLDOWN_SAMPLES,
            };
            Ok(Some(WakePipelineEvent::ClarificationNeeded))
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
        let mut capture_vad = EnergyVad::new(VadConfig::conversation())?;
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
            last_live_transcript: String::new(),
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
            if self.classifier.mentions_spoken_identity(&transcript) {
                let disposition = self.classifier.classify(&transcript)?.disposition;
                if disposition != WakeIntentDisposition::Addressed {
                    self.wake_window.clear();
                    if session_samples_left == 0 {
                        self.detector.reset(cancellation)?;
                        self.state = RuntimeState::Listening;
                    } else {
                        self.state = RuntimeState::Engaged {
                            vad: EnergyVad::new(VadConfig::conversation())?,
                            samples_left: session_samples_left,
                        };
                    }
                    return Ok((disposition == WakeIntentDisposition::Clarify)
                        .then_some(WakePipelineEvent::ClarificationNeeded));
                }
            } else if !self.warm_classifier.accepts(&transcript)? {
                self.wake_window.clear();
                self.state = RuntimeState::Engaged {
                    vad: EnergyVad::new(VadConfig::conversation())?,
                    samples_left: session_samples_left.max(1),
                };
                return Ok(None);
            }
            return self.complete_transcript(transcript, 0.0, cancellation);
        }
        self.wake_window.clear();
        if session_samples_left == 0 {
            self.detector.reset(cancellation)?;
            self.state = RuntimeState::Listening;
            Ok(Some(WakePipelineEvent::ConversationEnded))
        } else {
            self.state = RuntimeState::Engaged {
                vad: EnergyVad::new(VadConfig::conversation())?,
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
                    vad: EnergyVad::new(VadConfig::conversation())?,
                    samples: 0,
                    speech_started: false,
                };
                Ok(Some(WakePipelineEvent::WakeAccepted { score }))
            }
            ConversationDirective::Command => {
                self.wake_window.clear();
                self.state = RuntimeState::Engaged {
                    vad: EnergyVad::new(VadConfig::conversation())?,
                    samples_left: CONVERSATION_WINDOW_SAMPLES,
                };
                Ok(Some(WakePipelineEvent::CommandReady { transcript }))
            }
        }
    }
}

fn consume_session_time(samples_left: &mut usize, samples: usize, held: bool) {
    if !held {
        *samples_left = samples_left.saturating_sub(samples);
    }
}

fn tick_session(samples_left: &mut usize, chunk: &PcmChunk, held: bool) {
    consume_session_time(samples_left, chunk.samples().len(), held);
}

fn advance_command_capture(
    transcriber: &mut VoskTranscriber,
    max_command_samples: usize,
    chunk: &PcmChunk,
    cancellation: &CancellationToken,
    vad: &mut EnergyVad,
    samples: &mut usize,
    speech_started: &mut bool,
) -> Result<CommandCaptureProgress, AudioError> {
    *samples = samples.saturating_add(chunk.samples().len());
    transcriber.push(chunk, cancellation)?;
    match vad.observe(chunk) {
        VadDecision::SpeechStarted => *speech_started = true,
        VadDecision::Endpoint if *speech_started => return Ok(CommandCaptureProgress::Finish),
        VadDecision::Silence | VadDecision::Speech | VadDecision::Endpoint => {}
    }
    if (!*speech_started && *samples >= COMMAND_START_TIMEOUT_SAMPLES)
        || *samples >= max_command_samples
    {
        Ok(if *speech_started {
            CommandCaptureProgress::Finish
        } else {
            CommandCaptureProgress::TimedOut
        })
    } else {
        Ok(CommandCaptureProgress::Continue)
    }
}

#[allow(clippy::too_many_arguments)]
fn advance_engaged_capture(
    transcriber: &mut VoskTranscriber,
    phase: ConversationAudioPhase,
    max_command_samples: usize,
    chunk: &PcmChunk,
    cancellation: &CancellationToken,
    vad: &mut EnergyVad,
    samples: &mut usize,
    session_samples_left: &mut usize,
    last_live_transcript: &mut String,
) -> Result<EngagedCaptureProgress, AudioError> {
    *samples = samples.saturating_add(chunk.samples().len());
    tick_session(
        session_samples_left,
        chunk,
        phase != ConversationAudioPhase::Listening,
    );
    transcriber.push(chunk, cancellation)?;
    let vad_decision = vad.observe(chunk);
    if phase == ConversationAudioPhase::Speaking {
        let live = transcriber.live_transcript()?;
        if live != *last_live_transcript && has_barge_in_evidence(&live) {
            live.clone_into(last_live_transcript);
            return Ok(EngagedCaptureProgress::BargeIn(live));
        }
    }
    if vad_decision == VadDecision::Endpoint || *samples >= max_command_samples {
        Ok(EngagedCaptureProgress::Finished(*session_samples_left))
    } else {
        Ok(EngagedCaptureProgress::Continue)
    }
}

fn has_barge_in_evidence(transcript: &str) -> bool {
    transcript
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphanumeric))
        .take(2)
        .count()
        == 2
}

fn barge_in(transcript: String) -> WakePipelineEvent {
    WakePipelineEvent::PossibleBargeIn { transcript }
}

#[cfg(test)]
mod tests {
    use super::has_barge_in_evidence;

    #[test]
    fn barge_in_evidence_is_generic_and_requires_a_formed_partial() {
        assert!(!has_barge_in_evidence("wait"));
        assert!(has_barge_in_evidence("wait listen"));
        assert!(has_barge_in_evidence("explain supernovae"));
        assert!(has_barge_in_evidence("choose another route"));
    }
}

impl Drop for WakeCommandPipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}
