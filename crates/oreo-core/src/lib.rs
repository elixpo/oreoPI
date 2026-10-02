//! Deterministic foundations for the Oreo voice runtime.

mod cancellation;
mod config;
mod event;
mod runtime;

/// Verified Crumb building blocks used by Oreo's native agent boundary.
///
/// Keeping these re-exports behind one module prevents device and application
/// code from depending on the on-disk vendor layout.
pub mod crumb {
    pub use crumb_agent as agent;
    pub use crumb_harness as harness;
    pub use crumb_llm as llm;
    pub use crumb_pollinations as pollinations;
}

pub use cancellation::CancellationToken;
pub use config::{ConfigError, RuntimeConfig};
pub use event::{Affect, HarnessEvent, LocalIntent, Route, RuntimeEvent};
pub use runtime::{
    AssistantRuntime, CollectingSink, FakeHarness, Harness, OutputSink, ResponsePlan, RuntimeError,
    StdoutSink, TurnResult,
};
