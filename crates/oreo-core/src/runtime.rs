use std::error::Error;
use std::fmt;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

use crate::{
    Affect, CancellationToken, HarnessEvent, LocalIntent, Route, RuntimeConfig, RuntimeEvent,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponsePlan {
    pub speech: String,
    pub display: Option<String>,
    pub affect: Affect,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurnResult {
    pub route: Route,
    pub response: ResponsePlan,
}

pub trait Harness {
    /// Streams a bounded response through the supplied event callback.
    ///
    /// # Errors
    ///
    /// Returns a redacted runtime error for cancellation or harness failure.
    fn stream(
        &mut self,
        request: &str,
        cancellation: &CancellationToken,
        emit: &mut dyn FnMut(HarnessEvent) -> Result<(), RuntimeError>,
    ) -> Result<String, RuntimeError>;
}

pub trait OutputSink {
    /// Delivers one already-planned response.
    ///
    /// # Errors
    ///
    /// Returns a redacted error if the output backend is unavailable.
    fn deliver(&mut self, response: &ResponsePlan) -> Result<(), RuntimeError>;
}

#[derive(Debug, Default)]
pub struct FakeHarness;

impl Harness for FakeHarness {
    fn stream(
        &mut self,
        request: &str,
        cancellation: &CancellationToken,
        emit: &mut dyn FnMut(HarnessEvent) -> Result<(), RuntimeError>,
    ) -> Result<String, RuntimeError> {
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }
        emit(HarnessEvent::Started)?;
        let response = format!("Oreo heard: {request}");
        emit(HarnessEvent::TextDelta(response.clone()))?;
        emit(HarnessEvent::Completed)?;
        Ok(response)
    }
}

#[derive(Debug, Default)]
pub struct CollectingSink {
    pub responses: Vec<ResponsePlan>,
}

impl OutputSink for CollectingSink {
    fn deliver(&mut self, response: &ResponsePlan) -> Result<(), RuntimeError> {
        self.responses.push(response.clone());
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct StdoutSink;

impl OutputSink for StdoutSink {
    fn deliver(&mut self, response: &ResponsePlan) -> Result<(), RuntimeError> {
        println!("{}", response.speech);
        Ok(())
    }
}

pub struct AssistantRuntime<H, O> {
    config: RuntimeConfig,
    harness: H,
    output: O,
    event_sender: SyncSender<RuntimeEvent>,
    event_receiver: Receiver<RuntimeEvent>,
}

impl<H: Harness, O: OutputSink> AssistantRuntime<H, O> {
    /// Creates a runtime with one bounded event channel.
    ///
    /// # Errors
    ///
    /// Returns a configuration error when any runtime bound is invalid.
    pub fn new(config: RuntimeConfig, harness: H, output: O) -> Result<Self, RuntimeError> {
        config.validate()?;
        let (event_sender, event_receiver) = sync_channel(config.event_capacity);
        Ok(Self {
            config,
            harness,
            output,
            event_sender,
            event_receiver,
        })
    }

    /// Runs one deterministic request through routing, planning, and output.
    ///
    /// # Errors
    ///
    /// Rejects oversized input/output, cancellation, a saturated event queue,
    /// or a backend failure.
    pub fn handle(
        &mut self,
        request: &str,
        cancellation: &CancellationToken,
    ) -> Result<TurnResult, RuntimeError> {
        if request.len() > self.config.max_input_bytes {
            return Err(RuntimeError::InputTooLarge);
        }
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }

        let route = Route::classify(request);
        self.emit(RuntimeEvent::Routed(route))?;
        let speech = match route {
            Route::Local(intent) => local_response(intent).to_owned(),
            Route::Agent => {
                let sender = self.event_sender.clone();
                let mut emit = move |event| send_event(&sender, RuntimeEvent::Harness(event));
                self.harness.stream(request, cancellation, &mut emit)?
            }
        };
        if speech.len() > self.config.max_response_bytes {
            return Err(RuntimeError::ResponseTooLarge);
        }
        if cancellation.is_cancelled() {
            return Err(RuntimeError::Cancelled);
        }

        let response = ResponsePlan {
            speech,
            display: None,
            affect: Affect::Neutral,
        };
        self.emit(RuntimeEvent::ResponsePlanned {
            affect: response.affect,
        })?;
        self.output.deliver(&response)?;
        self.emit(RuntimeEvent::Delivered)?;
        Ok(TurnResult { route, response })
    }

    #[must_use]
    pub fn drain_events(&self) -> Vec<RuntimeEvent> {
        self.event_receiver.try_iter().collect()
    }

    #[must_use]
    pub fn output(&self) -> &O {
        &self.output
    }

    fn emit(&self, event: RuntimeEvent) -> Result<(), RuntimeError> {
        send_event(&self.event_sender, event)
    }
}

fn send_event(sender: &SyncSender<RuntimeEvent>, event: RuntimeEvent) -> Result<(), RuntimeError> {
    sender.try_send(event).map_err(|error| match error {
        TrySendError::Full(_) => RuntimeError::EventQueueFull,
        TrySendError::Disconnected(_) => RuntimeError::EventQueueDisconnected,
    })
}

const fn local_response(intent: LocalIntent) -> &'static str {
    match intent {
        LocalIntent::Pause => "Paused.",
        LocalIntent::Stop => "Stopped.",
        LocalIntent::VolumeUp | LocalIntent::VolumeDown => "",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    InvalidConfig(&'static str),
    InputTooLarge,
    ResponseTooLarge,
    Cancelled,
    EventQueueFull,
    EventQueueDisconnected,
    HarnessUnavailable,
    OutputUnavailable,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidConfig(message) => message,
            Self::InputTooLarge => "request exceeds its byte limit",
            Self::ResponseTooLarge => "response exceeds its byte limit",
            Self::Cancelled => "request cancelled",
            Self::EventQueueFull => "runtime event queue is full",
            Self::EventQueueDisconnected => "runtime event queue is unavailable",
            Self::HarnessUnavailable => "agent harness is unavailable",
            Self::OutputUnavailable => "output backend is unavailable",
        })
    }
}

impl Error for RuntimeError {}

impl From<crate::ConfigError> for RuntimeError {
    fn from(error: crate::ConfigError) -> Self {
        Self::InvalidConfig(error.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::{CollectingSink, FakeHarness, HarnessEvent, RuntimeEvent};

    use super::{AssistantRuntime, CancellationToken, Route, RuntimeConfig, RuntimeError};

    fn runtime() -> AssistantRuntime<FakeHarness, CollectingSink> {
        AssistantRuntime::new(RuntimeConfig::sbc(), FakeHarness, CollectingSink::default())
            .expect("valid fixture")
    }

    #[test]
    fn natural_request_completes_the_reference_path() {
        let mut runtime = runtime();
        let result = runtime
            .handle("What is running?", &CancellationToken::new())
            .expect("turn should complete");
        assert_eq!(result.route, Route::Agent);
        assert_eq!(result.response.speech, "Oreo heard: What is running?");
        assert_eq!(runtime.output().responses.len(), 1);
        assert!(
            runtime
                .drain_events()
                .contains(&RuntimeEvent::Harness(HarnessEvent::Completed))
        );
    }

    #[test]
    fn pause_never_calls_the_harness() {
        let mut runtime = runtime();
        let result = runtime
            .handle("pause", &CancellationToken::new())
            .expect("local turn should complete");
        assert!(matches!(result.route, Route::Local(_)));
        assert_eq!(result.response.speech, "Paused.");
        assert!(
            !runtime
                .drain_events()
                .iter()
                .any(|event| matches!(event, RuntimeEvent::Harness(_)))
        );
    }

    #[test]
    fn cancellation_stops_before_routing() {
        let mut runtime = runtime();
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert_eq!(
            runtime.handle("hello", &cancellation),
            Err(RuntimeError::Cancelled)
        );
        assert!(runtime.drain_events().is_empty());
    }

    #[test]
    fn oversized_input_is_rejected() {
        let mut config = RuntimeConfig::sbc();
        config.max_input_bytes = 3;
        let mut runtime = AssistantRuntime::new(config, FakeHarness, CollectingSink::default())
            .expect("valid fixture");
        assert_eq!(
            runtime.handle("four", &CancellationToken::new()),
            Err(RuntimeError::InputTooLarge)
        );
    }

    #[test]
    fn saturated_event_queue_fails_without_blocking() {
        let mut config = RuntimeConfig::sbc();
        config.event_capacity = 1;
        let mut runtime = AssistantRuntime::new(config, FakeHarness, CollectingSink::default())
            .expect("valid fixture");
        assert_eq!(
            runtime.handle("hello", &CancellationToken::new()),
            Err(RuntimeError::EventQueueFull)
        );
    }
}
