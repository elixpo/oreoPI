use crate::{AudioError, AudioErrorKind};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PipelinePhase {
    #[default]
    Idle,
    Capturing,
    Transcribing,
    Responding,
    Synthesizing,
    Playing,
    Cancelled,
    Faulted,
}

#[derive(Debug, Default)]
pub struct PushToTalkState {
    phase: PipelinePhase,
}

impl PushToTalkState {
    #[must_use]
    pub const fn phase(&self) -> PipelinePhase {
        self.phase
    }

    /// Begins capture for a new press.
    ///
    /// # Errors
    ///
    /// Rejects a press while another utterance is active.
    pub fn press(&mut self) -> Result<(), AudioError> {
        self.transition(PipelinePhase::Idle, PipelinePhase::Capturing)
    }

    /// Ends capture and starts transcription.
    ///
    /// # Errors
    ///
    /// Rejects release outside an active capture.
    pub fn release(&mut self) -> Result<(), AudioError> {
        self.transition(PipelinePhase::Capturing, PipelinePhase::Transcribing)
    }

    /// Marks transcript completion and response planning.
    ///
    /// # Errors
    ///
    /// Rejects completion outside transcription.
    pub fn transcript_ready(&mut self) -> Result<(), AudioError> {
        self.transition(PipelinePhase::Transcribing, PipelinePhase::Responding)
    }

    /// Marks the first speakable response chunk.
    ///
    /// # Errors
    ///
    /// Rejects synthesis outside response planning.
    pub fn response_ready(&mut self) -> Result<(), AudioError> {
        self.transition(PipelinePhase::Responding, PipelinePhase::Synthesizing)
    }

    /// Marks arrival of the first synthesized PCM chunk.
    ///
    /// # Errors
    ///
    /// Rejects playback outside synthesis.
    pub fn audio_ready(&mut self) -> Result<(), AudioError> {
        self.transition(PipelinePhase::Synthesizing, PipelinePhase::Playing)
    }

    /// Completes playback and returns to idle.
    ///
    /// # Errors
    ///
    /// Rejects completion outside playback.
    pub fn complete(&mut self) -> Result<(), AudioError> {
        self.transition(PipelinePhase::Playing, PipelinePhase::Idle)
    }

    pub fn cancel(&mut self) {
        self.phase = PipelinePhase::Cancelled;
    }

    pub fn fault(&mut self) {
        self.phase = PipelinePhase::Faulted;
    }

    pub fn reset(&mut self) {
        self.phase = PipelinePhase::Idle;
    }

    fn transition(
        &mut self,
        expected: PipelinePhase,
        next: PipelinePhase,
    ) -> Result<(), AudioError> {
        if self.phase != expected {
            return Err(AudioError::new(
                AudioErrorKind::InvalidTransition,
                "audio pipeline transition is invalid",
            ));
        }
        self.phase = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::AudioErrorKind;

    use super::{PipelinePhase, PushToTalkState};

    #[test]
    fn push_to_talk_happy_path_is_explicit() {
        let mut state = PushToTalkState::default();
        state.press().expect("capture starts");
        state.release().expect("capture ends");
        state.transcript_ready().expect("transcript completes");
        state.response_ready().expect("response starts");
        state.audio_ready().expect("audio starts");
        state.complete().expect("playback completes");
        assert_eq!(state.phase(), PipelinePhase::Idle);
    }

    #[test]
    fn invalid_transition_fails_without_changing_phase() {
        let mut state = PushToTalkState::default();
        let error = state.release().expect_err("idle cannot release");
        assert_eq!(error.kind, AudioErrorKind::InvalidTransition);
        assert_eq!(state.phase(), PipelinePhase::Idle);
    }

    #[test]
    fn cancellation_requires_explicit_reset() {
        let mut state = PushToTalkState::default();
        state.press().expect("capture starts");
        state.cancel();
        assert_eq!(state.phase(), PipelinePhase::Cancelled);
        assert!(state.press().is_err());
        state.reset();
        assert_eq!(state.phase(), PipelinePhase::Idle);
    }
}
