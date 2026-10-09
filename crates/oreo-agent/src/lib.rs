//! Oreo-owned boundary around the Crumb-derived native harness engine.
//!
//! Applications depend on this crate rather than the vendor directory. The
//! boundary owns Oreo's profile, session lifecycle, event vocabulary, and
//! redacted errors while the internal engine supplies the provider-neutral
//! execution loop.

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crumb_agent::{
    AgentMode, AgentSession, ApprovalBroker, CancellationToken, SessionId, SessionJournal,
    SteeringQueue, ToolHost, TurnStatus, export_session, list_sessions,
};
use crumb_harness::{
    Conversation, EventSink as CrumbEventSink, HarnessErrorKind, HarnessEvent, HarnessLimits,
    HarnessTurn, NativeHarness,
};
use crumb_llm::{LlmProvider, TokenUsage};

mod builtins;
mod capability;
mod mood;
mod routine;

pub use builtins::{DeviceStatus, register_device_status, register_memory_recall};
pub use capability::{
    ApprovalUi, Capability, CapabilityError, CapabilityLocation, CapabilityRegistry,
    ConfirmationPolicy, DenyApprovalUi,
};
pub use crumb_agent::SteeringAction;
pub use mood::{AffectState, DeliveryStyle};
pub use routine::{Routine, RoutineError, RoutineRunner, RoutineStep};

const EMBEDDED_STEERING_CUES: &str = include_str!("../../../config/voice-steering-cues.tsv");

/// Non-secret, bounded settings applied to every Oreo agent turn.
///
/// Persona and trusted context may contain private data, so this type does not
/// implement `Debug` and must not be written to logs.
#[derive(Clone)]
pub struct AgentProfile {
    pub model: String,
    pub persona: String,
    pub trusted_context: Vec<String>,
    pub max_output_tokens: Option<u32>,
    pub limits: HarnessLimits,
    pub mode: AgentMode,
}

impl AgentProfile {
    #[must_use]
    pub fn sbc(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            persona: String::new(),
            trusted_context: Vec::new(),
            max_output_tokens: Some(512),
            limits: HarnessLimits::default(),
            mode: AgentMode::Negotiate,
        }
    }

    /// Creates the tighter profile used by short spoken interactions.
    #[must_use]
    pub fn voice(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            persona: String::new(),
            trusted_context: Vec::new(),
            max_output_tokens: Some(256),
            limits: HarnessLimits {
                max_model_rounds: NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN),
                max_tool_calls: NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN),
                max_output_bytes: NonZeroUsize::new(8 * 1_024).unwrap_or(NonZeroUsize::MIN),
                max_history_turns: NonZeroUsize::new(4).unwrap_or(NonZeroUsize::MIN),
                turn_timeout: Duration::from_secs(45),
            },
            mode: AgentMode::Negotiate,
        }
    }

    /// Checks profile values before any session or provider work starts.
    ///
    /// # Errors
    ///
    /// Returns a redacted invalid-profile error for a missing model or a zero
    /// output-token limit.
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.model.trim().is_empty() {
            return Err(AgentError::new(
                AgentErrorKind::InvalidProfile,
                "agent model must not be empty",
            ));
        }
        if self.max_output_tokens == Some(0) {
            return Err(AgentError::new(
                AgentErrorKind::InvalidProfile,
                "agent output token limit must be positive",
            ));
        }
        Ok(())
    }
}

/// Oreo's stable, transient event vocabulary for one agent turn.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentEvent {
    Started,
    TextDelta(String),
    ToolRequested {
        id: String,
        name: String,
    },
    ToolFinished {
        id: String,
        name: String,
        success: bool,
    },
    Usage(TokenUsage),
    Completed,
}

pub trait EventSink {
    fn emit(&mut self, event: AgentEvent);
}

impl EventSink for Vec<AgentEvent> {
    fn emit(&mut self, event: AgentEvent) {
        self.push(event);
    }
}

/// Bounded result returned after a successful agent turn.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentResponse {
    pub text: String,
    pub usage: TokenUsage,
    pub model_rounds: usize,
    pub tool_calls: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurnSubmission {
    Queued,
    InterruptRequested,
}

/// Small local policy mapping conversational cues to harness steering actions.
pub struct VoiceTurnPolicy {
    queue_cues: Vec<String>,
    replace_cues: Vec<String>,
}

impl VoiceTurnPolicy {
    /// Loads reviewed steering cues without a model call.
    ///
    /// # Errors
    ///
    /// Rejects malformed, missing, or excessive embedded policy data.
    pub fn embedded() -> Result<Self, AgentError> {
        let mut queue_cues = Vec::new();
        let mut replace_cues = Vec::new();
        for line in EMBEDDED_STEERING_CUES.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((action, cue)) = line.split_once('\t') else {
                return Err(steering_policy_error());
            };
            let cue = normalize_cue(cue);
            if cue.is_empty() {
                return Err(steering_policy_error());
            }
            match action {
                "queue" => queue_cues.push(cue),
                "replace" => replace_cues.push(cue),
                _ => return Err(steering_policy_error()),
            }
        }
        if queue_cues.is_empty()
            || replace_cues.is_empty()
            || queue_cues.len().saturating_add(replace_cues.len()) > 32
        {
            return Err(steering_policy_error());
        }
        Ok(Self {
            queue_cues,
            replace_cues,
        })
    }

    /// Selects queue, replace, or immediate steering from local turn state.
    #[must_use]
    pub fn action(&self, transcript: &str, turn_active: bool) -> SteeringAction {
        if !turn_active {
            return SteeringAction::Queue;
        }
        let normalized = normalize_cue(transcript);
        if self
            .replace_cues
            .iter()
            .any(|cue| starts_with_cue(&normalized, cue))
        {
            SteeringAction::Replace
        } else if self
            .queue_cues
            .iter()
            .any(|cue| starts_with_cue(&normalized, cue))
        {
            SteeringAction::Queue
        } else {
            SteeringAction::Steer
        }
    }
}

fn normalize_cue(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '\'' {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn starts_with_cue(transcript: &str, cue: &str) -> bool {
    transcript == cue
        || transcript
            .strip_prefix(cue)
            .is_some_and(|remainder| remainder.starts_with(' '))
}

fn steering_policy_error() -> AgentError {
    AgentError::new(AgentErrorKind::Steering, "voice steering policy is invalid")
}

/// One bounded command removed from the transient turn queue.
pub struct QueuedAgentTurn {
    pub message: String,
    pub cancellation: CancellationToken,
}

/// Coordinates chained, steered, and replaced turns around one long-lived
/// [`OreoAgent`] without persisting voice transcripts.
pub struct AgentTurnController {
    queue: SteeringQueue,
    active: Option<ActiveAgentTurn>,
}

struct ActiveAgentTurn {
    cancellation: CancellationToken,
    message: String,
}

impl AgentTurnController {
    /// Creates a bounded transient turn controller.
    ///
    /// # Errors
    ///
    /// Rejects zero queue limits.
    pub fn new(max_messages: usize, max_bytes: usize) -> Result<Self, AgentError> {
        Ok(Self {
            queue: SteeringQueue::new(max_messages, max_bytes).map_err(|_| {
                AgentError::new(
                    AgentErrorKind::Steering,
                    "agent steering limits are invalid",
                )
            })?,
            active: None,
        })
    }

    /// Submits one follow-up policy and cancels the active turn for steer or
    /// replace. Queue preserves the active turn and runs afterward.
    ///
    /// # Errors
    ///
    /// Rejects empty, oversized, or over-capacity messages without mutation.
    pub fn submit(
        &mut self,
        action: SteeringAction,
        message: &str,
    ) -> Result<TurnSubmission, AgentError> {
        let preserved;
        let queued_message = if action == SteeringAction::Steer {
            if let Some(active) = &self.active {
                preserved = format!(
                    "Continue the active request while applying this correction.\nActive request: {}\nCorrection: {}",
                    active.message,
                    message.trim()
                );
                preserved.as_str()
            } else {
                message
            }
        } else {
            message
        };
        self.queue.submit(action, queued_message).map_err(|_| {
            AgentError::new(
                AgentErrorKind::Steering,
                "agent steering queue rejected the turn",
            )
        })?;
        let interrupts = matches!(action, SteeringAction::Steer | SteeringAction::Replace)
            && self.active.is_some();
        if interrupts && let Some(active) = &self.active {
            active.cancellation.cancel();
        }
        Ok(if interrupts {
            TurnSubmission::InterruptRequested
        } else {
            TurnSubmission::Queued
        })
    }

    /// Starts the next queued turn only when no prior turn is active.
    #[must_use]
    pub fn begin_next(&mut self) -> Option<QueuedAgentTurn> {
        if self.active.is_some() {
            return None;
        }
        let message = self.queue.pop()?;
        let cancellation = CancellationToken::default();
        self.active = Some(ActiveAgentTurn {
            cancellation: cancellation.clone(),
            message: message.clone(),
        });
        Some(QueuedAgentTurn {
            message,
            cancellation,
        })
    }

    /// Completes the current cancellation boundary and allows the next turn.
    pub fn finish_active(&mut self, cancellation: &CancellationToken) -> bool {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.cancellation.shares_signal_with(cancellation))
        {
            self.active = None;
            true
        } else {
            false
        }
    }

    pub fn cancel_all(&mut self) {
        if let Some(active) = self.active.take() {
            active.cancellation.cancel();
        }
        self.queue.clear();
    }

    #[must_use]
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }
}

/// Prompt-free metadata exposed by Oreo's user-owned memory inspector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemorySummary {
    pub id: String,
    pub archived: bool,
    pub mode: &'static str,
    pub started_at_ms: u64,
    pub last_event_at_ms: u64,
    pub turns: u32,
    pub last_status: Option<&'static str>,
}

/// Bounded inspection result for one redacted session journal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryInspection {
    pub summary: MemorySummary,
    pub retained_events: usize,
}

/// Lists prompt-free agent memory summaries newest-first.
///
/// # Errors
///
/// Returns a redacted session error when the journal root cannot be read.
pub fn list_memory(root: impl AsRef<Path>) -> Result<Vec<MemorySummary>, AgentError> {
    list_sessions(root.as_ref())
        .map(|summaries| summaries.iter().map(memory_summary).collect())
        .map_err(|_| AgentError::new(AgentErrorKind::Session, "agent memory is unavailable"))
}

/// Inspects one session without exposing prompts, responses, or tool payloads.
///
/// # Errors
///
/// Returns a redacted session error for an invalid, missing, or damaged journal.
pub fn inspect_memory(root: impl AsRef<Path>, id: &str) -> Result<MemoryInspection, AgentError> {
    let export = export_session(root.as_ref(), id)
        .map_err(|_| AgentError::new(AgentErrorKind::Session, "agent memory is unavailable"))?;
    Ok(MemoryInspection {
        summary: memory_summary(&export.summary),
        retained_events: export.events.len(),
    })
}

fn memory_summary(summary: &crumb_agent::SessionSummary) -> MemorySummary {
    MemorySummary {
        id: summary.id.as_str().to_owned(),
        archived: summary.archived,
        mode: match summary.mode {
            AgentMode::Auto => "auto",
            AgentMode::Negotiate => "negotiate",
            AgentMode::Plan => "plan",
        },
        started_at_ms: summary.started_at_ms,
        last_event_at_ms: summary.last_event_at_ms,
        turns: summary.turns,
        last_status: summary.last_status.map(|status| match status {
            TurnStatus::Complete => "complete",
            TurnStatus::Cancelled => "cancelled",
            TurnStatus::Failed => "failed",
            TurnStatus::LimitReached => "limit_reached",
        }),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentErrorKind {
    InvalidProfile,
    Session,
    Harness(HarnessErrorKind),
    Steering,
}

/// Redacted failure safe to expose to the local runtime and CLI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentError {
    pub kind: AgentErrorKind,
    message: &'static str,
}

impl AgentError {
    const fn new(kind: AgentErrorKind, message: &'static str) -> Self {
        Self { kind, message }
    }

    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        matches!(
            self.kind,
            AgentErrorKind::Harness(HarnessErrorKind::Cancelled)
        )
    }
}

impl fmt::Display for AgentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl Error for AgentError {}

/// One long-lived Oreo conversation and its secret-safe session journal.
pub struct OreoAgent {
    profile: AgentProfile,
    harness: NativeHarness,
    conversation: Conversation,
    session: AgentSession,
}

impl OreoAgent {
    /// Creates an agent without performing a provider network request.
    ///
    /// # Errors
    ///
    /// Returns a redacted error for invalid settings or if the local session
    /// metadata journal cannot be opened.
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        tools: ToolHost,
        approvals: Arc<dyn ApprovalBroker>,
        session_root: impl AsRef<Path>,
        session_id: &str,
        profile: AgentProfile,
    ) -> Result<Self, AgentError> {
        profile.validate()?;
        let id = SessionId::new(session_id)
            .map_err(|_| AgentError::new(AgentErrorKind::Session, "agent session id is invalid"))?;
        let root = session_root.as_ref();
        let journal = SessionJournal::open(root, &id).map_err(|_| {
            AgentError::new(
                AgentErrorKind::Session,
                "agent session journal is unavailable",
            )
        })?;
        let session =
            AgentSession::start(id, profile.mode, PathBuf::from(root), journal).map_err(|_| {
                AgentError::new(
                    AgentErrorKind::Session,
                    "agent session could not be started",
                )
            })?;
        let harness = NativeHarness::new(provider, tools, approvals, profile.limits);
        let conversation = harness.conversation();
        Ok(Self {
            profile,
            harness,
            conversation,
            session,
        })
    }

    /// Runs one bounded turn through the engine and translates events into
    /// Oreo's stable vocabulary.
    ///
    /// # Errors
    ///
    /// Returns only typed, redacted session or harness failures.
    pub async fn ask(
        &mut self,
        message: impl Into<String>,
        cancellation: &CancellationToken,
        sink: &mut dyn EventSink,
    ) -> Result<AgentResponse, AgentError> {
        let turn = HarnessTurn {
            model: self.profile.model.clone(),
            user_message: message.into(),
            persona: (!self.profile.persona.trim().is_empty())
                .then(|| self.profile.persona.clone()),
            trusted_context: self.profile.trusted_context.clone(),
            max_output_tokens: self.profile.max_output_tokens,
        };
        let mut bridge = EventBridge(sink);
        let outcome = self
            .harness
            .run_turn(
                &mut self.session,
                &mut self.conversation,
                turn,
                cancellation,
                &mut bridge,
            )
            .await
            .map_err(|error| {
                AgentError::new(
                    AgentErrorKind::Harness(error.kind),
                    "agent turn could not be completed",
                )
            })?;
        Ok(AgentResponse {
            text: outcome.text,
            usage: outcome.usage,
            model_rounds: outcome.model_rounds,
            tool_calls: outcome.tool_calls,
        })
    }

    pub fn reset_conversation(&mut self) {
        self.conversation = self.harness.conversation();
    }

    #[must_use]
    pub fn session_id(&self) -> &str {
        self.session.id().as_str()
    }
}

struct EventBridge<'a>(&'a mut dyn EventSink);

impl CrumbEventSink for EventBridge<'_> {
    fn emit(&mut self, event: HarnessEvent) {
        let event = match event {
            HarnessEvent::TurnStarted => AgentEvent::Started,
            HarnessEvent::TextDelta(delta) => AgentEvent::TextDelta(delta),
            HarnessEvent::ToolRequested { id, name } => AgentEvent::ToolRequested { id, name },
            HarnessEvent::ToolFinished { id, name, success } => {
                AgentEvent::ToolFinished { id, name, success }
            }
            HarnessEvent::Usage(usage) => AgentEvent::Usage(usage),
            HarnessEvent::TurnFinished => AgentEvent::Completed,
        };
        self.0.emit(event);
    }
}

/// Provider contracts used when constructing an [`OreoAgent`].
pub mod provider {
    pub use crumb_llm::{
        ChatEvent, ChatRequest, ChatStream, EmbeddingRequest, EmbeddingResponse, FinishReason,
        LlmProvider, ProviderError, ProviderErrorKind, ProviderFuture, TokenUsage, ToolCall,
    };
    pub use crumb_pollinations::{PollinationsConfig, PollinationsProvider, RetryPolicy};
}

/// Tool and approval contracts used when constructing an [`OreoAgent`].
pub mod tools {
    pub use crumb_agent::{
        ApprovalBroker, ApprovalDecision, ApprovalRequest, CancellationToken, ConfiguredApprovals,
        DenyAllApprovals, RiskClass, ToolDescriptor, ToolHandler, ToolHost, ToolOutput,
        ToolTransport,
    };
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crumb_agent::{CancellationToken, DenyAllApprovals, ToolHost};
    use crumb_llm::{
        ChatEvent, ChatRequest, ChatStream, EmbeddingRequest, EmbeddingResponse, FinishReason,
        LlmProvider, ModelInfo, ProviderError, ProviderErrorKind, ProviderFuture,
    };

    use super::{
        AgentEvent, AgentProfile, AgentTurnController, OreoAgent, SteeringAction, TurnSubmission,
        VoiceTurnPolicy, inspect_memory, list_memory,
    };

    struct FakeProvider {
        events: Mutex<VecDeque<ChatEvent>>,
        requests: Mutex<Vec<ChatRequest>>,
    }

    impl FakeProvider {
        fn answering(text: &str) -> Self {
            Self {
                events: Mutex::new(
                    vec![
                        ChatEvent::TextDelta(text.to_owned()),
                        ChatEvent::Finished(FinishReason::Stop),
                    ]
                    .into(),
                ),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl LlmProvider for FakeProvider {
        fn name(&self) -> &'static str {
            "fixture"
        }

        fn list_models(&self) -> ProviderFuture<'_, Vec<ModelInfo>> {
            Box::pin(future::ready(Ok(Vec::new())))
        }

        fn chat_stream(&self, request: ChatRequest) -> ProviderFuture<'_, Box<dyn ChatStream>> {
            self.requests.lock().expect("request lock").push(request);
            let events = self.events.lock().expect("event lock").drain(..).collect();
            let stream: Box<dyn ChatStream> = Box::new(FakeStream { events });
            Box::pin(future::ready(Ok(stream)))
        }

        fn embeddings(&self, _request: EmbeddingRequest) -> ProviderFuture<'_, EmbeddingResponse> {
            Box::pin(future::ready(Err(ProviderError::new(
                ProviderErrorKind::Other,
                "not implemented by fixture",
                false,
            ))))
        }
    }

    struct FakeStream {
        events: VecDeque<ChatEvent>,
    }

    impl ChatStream for FakeStream {
        fn next(&mut self) -> ProviderFuture<'_, Option<ChatEvent>> {
            Box::pin(future::ready(Ok(self.events.pop_front())))
        }
    }

    #[tokio::test]
    async fn oreo_profile_and_events_cross_the_adapter() {
        let provider = Arc::new(FakeProvider::answering("Hello from Oreo."));
        let mut profile = AgentProfile::sbc("fixture");
        profile.persona = "Warm, concise, and honest.".to_owned();
        profile.trusted_context = vec!["Locale: en-IN".to_owned()];
        let root = tempfile::tempdir().expect("session root");
        let mut agent = OreoAgent::new(
            provider.clone(),
            ToolHost::default(),
            Arc::new(DenyAllApprovals),
            root.path(),
            "oreo-test",
            profile,
        )
        .expect("agent starts");
        let mut events = Vec::new();

        let response = agent
            .ask("Say hello", &CancellationToken::default(), &mut events)
            .await
            .expect("turn succeeds");

        assert_eq!(response.text, "Hello from Oreo.");
        assert_eq!(agent.session_id(), "oreo-test");
        assert_eq!(events.first(), Some(&AgentEvent::Started));
        assert_eq!(events.last(), Some(&AgentEvent::Completed));
        let requests = provider.requests.lock().expect("request lock");
        assert_eq!(
            requests[0].messages[0].content,
            "Persona:\nWarm, concise, and honest."
        );
        assert_eq!(
            requests[0].messages[1].content,
            "Trusted context:\nLocale: en-IN"
        );
        drop(requests);

        let memories = list_memory(root.path()).expect("memory lists");
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].id, "oreo-test");
        assert_eq!(memories[0].turns, 1);
        assert_eq!(memories[0].last_status, Some("complete"));
        let inspection = inspect_memory(root.path(), "oreo-test").expect("memory inspects");
        assert_eq!(inspection.summary, memories[0]);
        assert_eq!(inspection.retained_events, 3);
    }

    #[test]
    fn invalid_profile_fails_before_session_creation() {
        let profile = AgentProfile::sbc(" ");
        let error = profile.validate().expect_err("blank model is invalid");
        assert_eq!(error.to_string(), "agent model must not be empty");
    }

    #[test]
    fn voice_profile_bounds_spoken_turn_cost() {
        let profile = AgentProfile::voice("fixture");
        assert_eq!(profile.max_output_tokens, Some(256));
        assert_eq!(profile.limits.max_model_rounds.get(), 4);
        assert_eq!(profile.limits.max_tool_calls.get(), 4);
        assert_eq!(profile.limits.max_output_bytes.get(), 8 * 1_024);
        assert_eq!(profile.limits.max_history_turns.get(), 4);
        assert_eq!(profile.limits.turn_timeout, Duration::from_secs(45));
    }

    #[test]
    fn turn_controller_chains_steers_and_replaces_with_shared_cancellation() {
        let mut controller = AgentTurnController::new(3, 128).expect("controller starts");
        assert_eq!(
            controller
                .submit(SteeringAction::Queue, "first")
                .expect("first turn queues"),
            TurnSubmission::Queued
        );
        let first = controller.begin_next().expect("first turn starts");
        assert!(controller.is_active());

        assert_eq!(
            controller
                .submit(SteeringAction::Queue, "later")
                .expect("follow-up queues"),
            TurnSubmission::Queued
        );
        assert!(!first.cancellation.is_cancelled());
        assert_eq!(
            controller
                .submit(SteeringAction::Steer, "actually do this")
                .expect("steer queues first"),
            TurnSubmission::InterruptRequested
        );
        assert!(first.cancellation.is_cancelled());
        assert!(controller.finish_active(&first.cancellation));

        let steered = controller.begin_next().expect("steer starts next");
        assert!(steered.message.contains("Active request: first"));
        assert!(steered.message.contains("Correction: actually do this"));
        assert!(controller.finish_active(&steered.cancellation));
        let chained = controller.begin_next().expect("queued turn follows");
        assert_eq!(chained.message, "later");

        controller
            .submit(SteeringAction::Replace, "replacement")
            .expect("replacement interrupts");
        assert!(chained.cancellation.is_cancelled());
        assert!(controller.finish_active(&chained.cancellation));
        assert_eq!(
            controller
                .begin_next()
                .expect("replacement remains")
                .message,
            "replacement"
        );
    }

    #[test]
    fn voice_policy_queues_replaces_and_steers_contextually() {
        let policy = VoiceTurnPolicy::embedded().expect("policy loads");
        assert_eq!(
            policy.action("ordinary first command", false),
            SteeringAction::Queue
        );
        assert_eq!(
            policy.action("also check the weather", true),
            SteeringAction::Queue
        );
        assert_eq!(
            policy.action("Never mind, check the door instead", true),
            SteeringAction::Replace
        );
        assert_eq!(
            policy.action("stop and tell me the time", true),
            SteeringAction::Steer
        );
    }
}
