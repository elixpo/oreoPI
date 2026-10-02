//! Deterministic foundations for the Oreo voice runtime.

mod cancellation;
mod config;
mod event;
mod runtime;

pub use cancellation::CancellationToken;
pub use config::{ConfigError, RuntimeConfig};
pub use event::{Affect, HarnessEvent, LocalIntent, Route, RuntimeEvent};
pub use runtime::{
    AssistantRuntime, CollectingSink, FakeHarness, Harness, OutputSink, ResponsePlan, RuntimeError,
    StdoutSink, TurnResult,
};
