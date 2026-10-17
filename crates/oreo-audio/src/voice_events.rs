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
