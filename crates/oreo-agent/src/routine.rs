//! Bounded routine execution through the existing capability policy.

use std::error::Error;
use std::fmt;

use crumb_agent::{AgentMode, ApprovalBroker, CancellationToken, ToolHost, ToolOutput};
use serde_json::Value;

const MAX_ROUTINE_STEPS: usize = 16;
const MAX_ROUTINE_BYTES: usize = 16 * 1_024;

#[derive(Clone, Debug, PartialEq)]
pub struct RoutineStep {
    pub capability: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Routine {
    pub id: String,
    pub steps: Vec<RoutineStep>,
}

impl Routine {
    /// Creates a data-only routine. Shell commands and arbitrary transports are
    /// impossible because every step names a registered capability.
    ///
    /// # Errors
    ///
    /// Rejects invalid identifiers, empty/excessive steps, non-object
    /// arguments, or an excessive encoded size.
    pub fn new(id: impl Into<String>, steps: Vec<RoutineStep>) -> Result<Self, RoutineError> {
        let id = id.into();
        if id.is_empty()
            || id.len() > 64
            || !id.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '-' | '_')
            })
            || steps.is_empty()
            || steps.len() > MAX_ROUTINE_STEPS
            || steps.iter().any(|step| {
                step.capability.is_empty()
                    || step.capability.len() > 64
                    || !step.arguments.is_object()
            })
        {
            return Err(RoutineError::new("routine definition is invalid"));
        }
        let encoded =
            serde_json::to_vec(&steps.iter().map(|step| &step.arguments).collect::<Vec<_>>())
                .map_err(|_| RoutineError::new("routine definition is invalid"))?;
        if encoded.len() > MAX_ROUTINE_BYTES {
            return Err(RoutineError::new("routine definition is too large"));
        }
        Ok(Self { id, steps })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RoutineRunner;

impl RoutineRunner {
    /// Executes steps sequentially through the normal tool host and approval
    /// broker. Cancellation and any denied/failed step stop the routine.
    ///
    /// # Errors
    ///
    /// Returns a redacted error when a step is cancelled, denied, or fails.
    pub fn run(
        self,
        routine: &Routine,
        tools: &ToolHost,
        approvals: &dyn ApprovalBroker,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ToolOutput>, RoutineError> {
        let mut outputs = Vec::with_capacity(routine.steps.len());
        for step in &routine.steps {
            let output = tools
                .call(
                    &step.capability,
                    &step.arguments,
                    AgentMode::Negotiate,
                    approvals,
                    cancellation,
                )
                .map_err(|_| RoutineError::new("routine step was not completed"))?;
            if output.is_error {
                return Err(RoutineError::new("routine step reported an error"));
            }
            outputs.push(output);
        }
        Ok(outputs)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutineError {
    message: &'static str,
}

impl RoutineError {
    const fn new(message: &'static str) -> Self {
        Self { message }
    }
}

impl fmt::Display for RoutineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl Error for RoutineError {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crumb_agent::{
        ApprovalDecision, ApprovalRequest, RiskClass, ToolDescriptor, ToolHandler, ToolTransport,
    };
    use serde_json::json;

    use super::*;

    struct FixtureHandler;

    struct Permit;

    impl ApprovalBroker for Permit {
        fn decide(
            &self,
            _request: &ApprovalRequest,
            _arguments: &Value,
            _cancellation: &CancellationToken,
        ) -> ApprovalDecision {
            ApprovalDecision::AllowOnce
        }
    }

    impl ToolHandler for FixtureHandler {
        fn call(
            &self,
            _arguments: &Value,
            _cancellation: &CancellationToken,
        ) -> anyhow::Result<ToolOutput> {
            Ok(ToolOutput::text("done"))
        }
    }

    #[test]
    fn routine_uses_tool_policy_for_every_step() {
        let mut tools = ToolHost::default();
        tools
            .register(
                ToolDescriptor {
                    name: "status".to_owned(),
                    description: "Read status".to_owned(),
                    input_schema: json!({"type": "object"}),
                    risk: RiskClass::ReadOnly,
                    transport: ToolTransport::Native,
                },
                Arc::new(FixtureHandler),
            )
            .expect("tool registers");
        let routine = Routine::new(
            "morning",
            vec![RoutineStep {
                capability: "status".to_owned(),
                arguments: json!({}),
            }],
        )
        .expect("routine validates");
        let outputs = RoutineRunner
            .run(&routine, &tools, &Permit, &CancellationToken::default())
            .expect("routine runs");
        assert_eq!(outputs, vec![ToolOutput::text("done")]);
    }

    #[test]
    fn routine_rejects_shell_shaped_identity_and_unbounded_steps() {
        assert!(Routine::new("morning; shutdown", Vec::new()).is_err());
        let steps = (0..=MAX_ROUTINE_STEPS)
            .map(|_| RoutineStep {
                capability: "status".to_owned(),
                arguments: json!({}),
            })
            .collect();
        assert!(Routine::new("too-long", steps).is_err());
    }
}
