//! Oreo-owned boundary around the vendored Crumb native harness.
//!
//! Applications depend on this crate rather than the vendor directory. The
//! boundary owns Oreo's profile, session lifecycle, event vocabulary, and
//! redacted errors while Crumb supplies the provider-neutral execution loop.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crumb_agent::{
    AgentMode, AgentSession, ApprovalBroker, CancellationToken, SessionId, SessionJournal, ToolHost,
};
use crumb_harness::{
    Conversation, EventSink as CrumbEventSink, HarnessErrorKind, HarnessEvent, HarnessLimits,
    HarnessTurn, NativeHarness,
};
use crumb_llm::{LlmProvider, TokenUsage};

mod capability;

pub use capability::{
    ApprovalUi, Capability, CapabilityError, CapabilityLocation, CapabilityRegistry,
    ConfirmationPolicy, DenyApprovalUi,
};

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
pub enum AgentErrorKind {
    InvalidProfile,
    Session,
    Harness(HarnessErrorKind),
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

    /// Runs one bounded turn through Crumb and translates events into Oreo's
    /// stable vocabulary.
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

    use crumb_agent::{CancellationToken, DenyAllApprovals, ToolHost};
    use crumb_llm::{
        ChatEvent, ChatRequest, ChatStream, EmbeddingRequest, EmbeddingResponse, FinishReason,
        LlmProvider, ModelInfo, ProviderError, ProviderErrorKind, ProviderFuture,
    };

    use super::{AgentEvent, AgentProfile, OreoAgent};

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
    }

    #[test]
    fn invalid_profile_fails_before_session_creation() {
        let profile = AgentProfile::sbc(" ");
        let error = profile.validate().expect_err("blank model is invalid");
        assert_eq!(error.to_string(), "agent model must not be empty");
    }
}
