use oreo_core::CancellationToken;

use crate::{
    AmbientAttentionDecision, AmbientAttentionGate, AudioError, AudioErrorKind, AudioLimits,
    ConversationAudioPhase, EnergyVad, PcmChunk, STT_FORMAT, StreamingTranscriber, VadConfig,
    VadDecision, VoskTranscriber, WakePipelineEvent,
};

const CONVERSATION_WINDOW_SAMPLES: usize = 30 * 16_000;
const MIN_TRANSCRIPT_CONFIDENCE: f32 = 0.35;

/// Continuous, fully local speech segmentation and contextual attention gate.
///
/// The Vosk model is reused across utterances. A completed transcript can only
/// leave this component after the local attention model accepts it.
pub struct AmbientVoicePipeline {
    transcriber: VoskTranscriber,
    attention: AmbientAttentionGate,
    attention_name: String,
    vad: EnergyVad,
    limits: AudioLimits,
    samples: usize,
    max_samples: usize,
    session_samples_left: usize,
    conversation_phase: ConversationAudioPhase,
    last_live_transcript: String,
    latest_partial: String,
    latest_final: Option<String>,
    latest_attention_match: bool,
    barge_in_confirmed: bool,
}

impl AmbientVoicePipeline {
    /// Creates the always-listening local pipeline.
    ///
    /// # Errors
    ///
    /// Rejects invalid bounds, attention names, or embedded classifiers.
    pub fn new(
        mut transcriber: VoskTranscriber,
        limits: AudioLimits,
        attention_name: &str,
    ) -> Result<Self, AudioError> {
        let limits = limits.validate()?;
        let max_samples = usize::from(limits.max_capture_seconds)
            .checked_mul(16_000)
            .ok_or_else(|| {
                AudioError::new(
                    AudioErrorKind::InvalidConfig,
                    "ambient capture limit is invalid",
                )
            })?;
        transcriber.configure_attention_spotter(attention_name)?;
        transcriber.begin(STT_FORMAT)?;
        Ok(Self {
            transcriber,
            attention: AmbientAttentionGate::new(attention_name)?,
            attention_name: attention_name.to_ascii_lowercase(),
            vad: EnergyVad::new(VadConfig::conversation())?,
            limits,
            samples: 0,
            max_samples,
            session_samples_left: 0,
            conversation_phase: ConversationAudioPhase::Listening,
            last_live_transcript: String::new(),
            latest_partial: String::new(),
            latest_final: None,
            latest_attention_match: false,
            barge_in_confirmed: false,
        })
    }

    /// Holds session expiry while the agent is thinking or speaking, while
    /// leaving microphone inference active for steering and barge-in.
    pub fn set_conversation_phase(&mut self, phase: ConversationAudioPhase) {
        self.conversation_phase = phase;
    }

    /// Marks the current utterance as a user-confirmed interruption.
    #[must_use]
    pub fn confirm_barge_in(&mut self) -> bool {
        if self.conversation_phase == ConversationAudioPhase::Speaking
            && !self.latest_partial.trim().is_empty()
        {
            self.barge_in_confirmed = true;
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn latest_partial(&self) -> &str {
        &self.latest_partial
    }

    pub fn take_final_transcript(&mut self) -> Option<String> {
        self.latest_final.take()
    }

    pub fn take_attention_match(&mut self) -> bool {
        std::mem::take(&mut self.latest_attention_match)
    }

    /// Processes one microphone chunk. All inference before `CommandReady` is
    /// local and raw PCM remains bounded to the current Vosk utterance.
    ///
    /// # Errors
    ///
    /// Fails closed on format, cancellation, Vosk, or classifier errors.
    pub fn process(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        if chunk.format() != STT_FORMAT {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "ambient voice runtime requires 16 kHz mono PCM",
            ));
        }
        if cancellation.is_cancelled() {
            self.transcriber.abort();
            return Err(AudioError::new(
                AudioErrorKind::Cancelled,
                "ambient voice runtime was cancelled",
            ));
        }

        self.samples = self.samples.saturating_add(chunk.samples().len());
        if self.attention.is_attentive()
            && self.conversation_phase == ConversationAudioPhase::Listening
        {
            self.session_samples_left = self
                .session_samples_left
                .saturating_sub(chunk.samples().len());
        }
        self.transcriber.push(chunk, cancellation)?;
        let vad = self.vad.observe(chunk);
        let live = self.transcriber.live_transcript()?;
        live.clone_into(&mut self.latest_partial);

        if self.conversation_phase == ConversationAudioPhase::Speaking
            && live != self.last_live_transcript
            && formed_partial(&live)
            && self.attention.accepts_warm_partial(&live)?
        {
            live.clone_into(&mut self.last_live_transcript);
            return Ok(Some(WakePipelineEvent::PossibleBargeIn {
                transcript: live,
            }));
        }

        if vad == VadDecision::Endpoint || self.samples >= self.max_samples {
            return self.finish_utterance(cancellation);
        }
        if vad == VadDecision::Silence
            && self.attention.is_attentive()
            && self.session_samples_left == 0
        {
            self.attention.sleep();
            return Ok(Some(WakePipelineEvent::ConversationEnded));
        }
        Ok(None)
    }

    pub fn shutdown(&mut self) {
        self.transcriber.abort();
        self.latest_partial.clear();
        self.latest_final = None;
        self.latest_attention_match = false;
    }

    fn finish_utterance(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        let transcript = self.transcriber.finish(cancellation)?;
        let confidence = self.transcriber.confidence();
        let attention_detected = self.transcriber.attention_detected();
        self.latest_attention_match = attention_detected;
        self.transcriber.begin(STT_FORMAT)?;
        self.vad = EnergyVad::new(VadConfig::conversation())?;
        self.samples = 0;
        self.last_live_transcript.clear();
        self.latest_partial.clear();
        self.latest_final = (!transcript.trim().is_empty()).then(|| transcript.clone());

        if (transcript.trim().is_empty() && !attention_detected)
            || transcript.len() > self.limits.max_transcript_bytes
            || confidence
                .is_some_and(|value| value.words != 0 && value.mean < MIN_TRANSCRIPT_CONFIDENCE)
        {
            self.barge_in_confirmed = false;
            return Ok(None);
        }

        let contextual_transcript =
            if attention_detected && !contains_word(&transcript, &self.attention_name) {
                if transcript.trim().is_empty() {
                    self.attention_name.clone()
                } else {
                    format!("{transcript} {}", self.attention_name)
                }
            } else {
                transcript.clone()
            };
        let decision = if self.barge_in_confirmed {
            self.barge_in_confirmed = false;
            AmbientAttentionDecision::Command
        } else {
            self.attention.assess(&contextual_transcript)?
        };
        match decision {
            AmbientAttentionDecision::Ignore => Ok(None),
            AmbientAttentionDecision::AttentionRaised => {
                self.session_samples_left = CONVERSATION_WINDOW_SAMPLES;
                Ok(Some(WakePipelineEvent::WakeAccepted { score: 0.0 }))
            }
            AmbientAttentionDecision::Command => {
                self.session_samples_left = CONVERSATION_WINDOW_SAMPLES;
                Ok(Some(WakePipelineEvent::CommandReady { transcript }))
            }
            AmbientAttentionDecision::Sleep => {
                self.session_samples_left = 0;
                Ok(Some(WakePipelineEvent::ConversationEnded))
            }
        }
    }
}

fn formed_partial(transcript: &str) -> bool {
    transcript
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphanumeric))
        .take(2)
        .count()
        == 2
}

fn contains_word(transcript: &str, expected: &str) -> bool {
    transcript
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|word| word.eq_ignore_ascii_case(expected))
}

impl Drop for AmbientVoicePipeline {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::{contains_word, formed_partial};

    #[test]
    fn interruption_requires_a_formed_partial() {
        assert!(!formed_partial("wait"));
        assert!(formed_partial("wait listen"));
    }

    #[test]
    fn attention_word_matching_does_not_accept_substrings() {
        assert!(contains_word("hello Orion", "orion"));
        assert!(!contains_word("orionis is unrelated", "orion"));
    }
}
