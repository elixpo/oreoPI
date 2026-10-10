use oreo_core::CancellationToken;
use std::collections::VecDeque;

use crate::{
    AcousticAttentionDetector, AmbientAttentionDecision, AmbientAttentionGate, AmbientTranscriber,
    AudioError, AudioErrorKind, AudioLimits, ConversationAudioPhase, EnergyVad, PcmChunk,
    STT_FORMAT, VadConfig, VadDecision, WakePipelineEvent,
};

const CONVERSATION_WINDOW_SAMPLES: usize = 30 * 16_000;
const MIN_TRANSCRIPT_CONFIDENCE: f32 = 0.35;
const WARM_PRE_ROLL_CHUNKS: usize = 10;
const COLD_PRE_ROLL_CHUNKS: usize = 150;

#[derive(Clone, Copy, Eq, PartialEq)]
enum UtteranceState {
    Dormant,
    Active,
}

/// Continuous, fully local speech segmentation and contextual attention gate.
///
/// The local STT model is reused across utterances. A completed transcript can only
/// leave this component after the local attention model accepts it.
pub struct AmbientVoicePipeline {
    transcriber: Box<dyn AmbientTranscriber>,
    detector: Box<dyn AcousticAttentionDetector>,
    attention: AmbientAttentionGate,
    attention_name: String,
    vad: EnergyVad,
    limits: AudioLimits,
    samples: usize,
    max_samples: usize,
    pre_roll: VecDeque<PcmChunk>,
    utterance_state: UtteranceState,
    session_samples_left: usize,
    conversation_phase: ConversationAudioPhase,
    last_live_transcript: String,
    latest_partial: String,
    latest_final: Option<String>,
    latest_final_accepted: bool,
    latest_attention_match: bool,
    barge_in_confirmed: bool,
    cold_attention_score: Option<f32>,
}

impl AmbientVoicePipeline {
    /// Creates the always-listening local pipeline.
    ///
    /// # Errors
    ///
    /// Rejects invalid bounds, attention names, or embedded classifiers.
    pub fn new(
        transcriber: impl AmbientTranscriber + 'static,
        detector: impl AcousticAttentionDetector + 'static,
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
        let mut transcriber: Box<dyn AmbientTranscriber> = Box::new(transcriber);
        transcriber.configure_attention_spotter(attention_name)?;
        Ok(Self {
            transcriber,
            detector: Box::new(detector),
            attention: AmbientAttentionGate::new(attention_name)?,
            attention_name: attention_name.to_ascii_lowercase(),
            vad: EnergyVad::new(VadConfig::conversation())?,
            limits,
            samples: 0,
            max_samples,
            pre_roll: VecDeque::with_capacity(COLD_PRE_ROLL_CHUNKS),
            utterance_state: UtteranceState::Dormant,
            session_samples_left: 0,
            conversation_phase: ConversationAudioPhase::Listening,
            last_live_transcript: String::new(),
            latest_partial: String::new(),
            latest_final: None,
            latest_final_accepted: false,
            latest_attention_match: false,
            barge_in_confirmed: false,
            cold_attention_score: None,
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

    #[must_use]
    pub const fn latest_final_was_accepted(&self) -> bool {
        self.latest_final_accepted
    }

    pub fn take_attention_match(&mut self) -> bool {
        std::mem::take(&mut self.latest_attention_match)
    }

    /// Processes one microphone chunk. All inference before `CommandReady` is
    /// local and raw PCM remains bounded to a short pre-roll and the current
    /// utterance. The larger STT backend remains idle until VAD confirms speech.
    ///
    /// # Errors
    ///
    /// Fails closed on format, cancellation, STT, or classifier errors.
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

        if self.attention.is_attentive()
            && self.conversation_phase == ConversationAudioPhase::Listening
        {
            self.session_samples_left = self
                .session_samples_left
                .saturating_sub(chunk.samples().len());
        }
        let vad = self.vad.observe(chunk);
        let attentive = self.attention.is_attentive();
        let cold_candidate = if attentive {
            None
        } else {
            self.detect_cold_candidate(vad, chunk, cancellation)?
        };
        if let Some(score) = cold_candidate {
            self.cold_attention_score = Some(
                self.cold_attention_score
                    .map_or(score, |previous| previous.max(score)),
            );
        }

        if self.utterance_state == UtteranceState::Active {
            self.samples = self.samples.saturating_add(chunk.samples().len());
            self.transcriber.push(chunk, cancellation)?;
        } else {
            self.pre_roll.push_back(chunk.clone());
            if self.pre_roll.len() > COLD_PRE_ROLL_CHUNKS {
                self.pre_roll.pop_front();
            }
            if (!attentive && cold_candidate.is_none())
                || (attentive && vad != VadDecision::SpeechStarted)
            {
                if vad == VadDecision::Silence
                    && self.attention.is_attentive()
                    && self.session_samples_left == 0
                {
                    self.attention.sleep();
                    return Ok(Some(WakePipelineEvent::ConversationEnded));
                }
                return Ok(None);
            }

            self.transcriber.begin(STT_FORMAT)?;
            self.utterance_state = UtteranceState::Active;
            if attentive {
                while self.pre_roll.len() > WARM_PRE_ROLL_CHUNKS {
                    self.pre_roll.pop_front();
                }
            }
            while let Some(buffered) = self.pre_roll.pop_front() {
                self.samples = self.samples.saturating_add(buffered.samples().len());
                self.transcriber.push(&buffered, cancellation)?;
            }
        }

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
        Ok(None)
    }

    fn detect_cold_candidate(
        &mut self,
        vad: VadDecision,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<Option<f32>, AudioError> {
        match vad {
            VadDecision::Silence => Ok(None),
            VadDecision::Endpoint => {
                self.detector.reset(cancellation)?;
                Ok(None)
            }
            VadDecision::Speech => self.detector.push(chunk, cancellation),
            VadDecision::SpeechStarted => {
                let buffered: Vec<_> = self
                    .pre_roll
                    .iter()
                    .rev()
                    .take(WARM_PRE_ROLL_CHUNKS.saturating_sub(1))
                    .rev()
                    .cloned()
                    .collect();
                let mut strongest = None::<f32>;
                for buffered_chunk in &buffered {
                    if let Some(score) = self.detector.push(buffered_chunk, cancellation)? {
                        strongest = Some(strongest.map_or(score, |value| value.max(score)));
                    }
                }
                if let Some(score) = self.detector.push(chunk, cancellation)? {
                    strongest = Some(strongest.map_or(score, |value| value.max(score)));
                }
                Ok(strongest)
            }
        }
    }

    pub fn shutdown(&mut self) {
        self.transcriber.suspend();
        self.detector.shutdown();
        self.utterance_state = UtteranceState::Dormant;
        self.pre_roll.clear();
        self.samples = 0;
        self.latest_partial.clear();
        self.latest_final = None;
        self.latest_final_accepted = false;
        self.latest_attention_match = false;
    }

    fn finish_utterance(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakePipelineEvent>, AudioError> {
        let transcript = self.transcriber.finish(cancellation)?;
        let confidence = self.transcriber.confidence();
        let attention_detected =
            self.cold_attention_score.take().is_some() || self.transcriber.attention_detected();
        self.transcriber.suspend();
        self.latest_attention_match = attention_detected;
        self.detector.reset(cancellation)?;
        self.utterance_state = UtteranceState::Dormant;
        self.pre_roll.clear();
        self.vad = EnergyVad::new(VadConfig::conversation())?;
        self.samples = 0;
        self.last_live_transcript.clear();
        self.latest_partial.clear();
        self.latest_final = (!transcript.trim().is_empty()).then(|| transcript.clone());
        self.latest_final_accepted = false;

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
                self.latest_final_accepted = true;
                self.session_samples_left = CONVERSATION_WINDOW_SAMPLES;
                Ok(Some(WakePipelineEvent::WakeAccepted { score: 0.0 }))
            }
            AmbientAttentionDecision::Command => {
                self.latest_final_accepted = true;
                self.session_samples_left = CONVERSATION_WINDOW_SAMPLES;
                Ok(Some(WakePipelineEvent::CommandReady { transcript }))
            }
            AmbientAttentionDecision::Sleep => {
                self.latest_final_accepted = true;
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
    use std::sync::{Arc, Mutex};

    use oreo_core::CancellationToken;

    use crate::{
        AcousticAttentionDetector, AmbientTranscriber, AudioError, AudioFormat, AudioLimits,
        PcmChunk, STT_FORMAT, StreamingTranscriber, TranscriptionConfidence,
    };

    use super::{AmbientVoicePipeline, contains_word, formed_partial};

    #[derive(Default)]
    struct TranscriberStats {
        begins: usize,
        pushes: usize,
    }

    struct FakeTranscriber {
        stats: Arc<Mutex<TranscriberStats>>,
    }

    struct FakeDetector {
        frames_until_candidate: usize,
    }

    impl AcousticAttentionDetector for FakeDetector {
        fn push(
            &mut self,
            _chunk: &PcmChunk,
            _cancellation: &CancellationToken,
        ) -> Result<Option<f32>, AudioError> {
            self.frames_until_candidate = self.frames_until_candidate.saturating_sub(1);
            Ok((self.frames_until_candidate == 0).then_some(0.9))
        }

        fn reset(&mut self, _cancellation: &CancellationToken) -> Result<(), AudioError> {
            Ok(())
        }

        fn shutdown(&mut self) {}
    }

    impl StreamingTranscriber for FakeTranscriber {
        fn begin(&mut self, _format: AudioFormat) -> Result<(), AudioError> {
            self.stats.lock().expect("stats").begins += 1;
            Ok(())
        }

        fn push(
            &mut self,
            _chunk: &PcmChunk,
            _cancellation: &CancellationToken,
        ) -> Result<(), AudioError> {
            self.stats.lock().expect("stats").pushes += 1;
            Ok(())
        }

        fn finish(&mut self, _cancellation: &CancellationToken) -> Result<String, AudioError> {
            Ok(String::new())
        }
    }

    impl AmbientTranscriber for FakeTranscriber {
        fn configure_attention_spotter(&mut self, _name: &str) -> Result<(), AudioError> {
            Ok(())
        }

        fn live_transcript(&mut self) -> Result<String, AudioError> {
            Ok(String::new())
        }

        fn confidence(&self) -> Option<TranscriptionConfidence> {
            None
        }

        fn attention_detected(&self) -> bool {
            false
        }
    }

    fn chunk(amplitude: i16) -> PcmChunk {
        PcmChunk::new(STT_FORMAT, vec![amplitude; 320]).expect("PCM chunk")
    }

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

    #[test]
    fn dormant_pipeline_does_not_run_stt_until_speech_is_confirmed() {
        let stats = Arc::new(Mutex::new(TranscriberStats::default()));
        let mut pipeline = AmbientVoicePipeline::new(
            FakeTranscriber {
                stats: Arc::clone(&stats),
            },
            FakeDetector {
                frames_until_candidate: 10,
            },
            AudioLimits::sbc(),
            "orion",
        )
        .expect("pipeline");
        let cancellation = CancellationToken::new();

        for _ in 0..20 {
            pipeline
                .process(&chunk(0), &cancellation)
                .expect("silent frame");
        }
        for _ in 0..3 {
            pipeline
                .process(&chunk(1_000), &cancellation)
                .expect("unconfirmed voice frame");
        }
        {
            let stats = stats.lock().expect("stats");
            assert_eq!(stats.begins, 0);
            assert_eq!(stats.pushes, 0);
        }

        pipeline
            .process(&chunk(1_000), &cancellation)
            .expect("confirmed voice frame");
        let stats = stats.lock().expect("stats");
        assert_eq!(stats.begins, 1);
        assert_eq!(stats.pushes, 24);
    }
}
