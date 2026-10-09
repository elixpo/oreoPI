//! Small, deterministic capabilities available on laptops and SBCs.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crumb_agent::{CancellationToken, RiskClass, ToolHandler, ToolOutput};
use oreo_state::{MemoryScope, StateLimits, StateStore, SystemClock};
use serde_json::{Value, json};

use crate::{
    Capability, CapabilityError, CapabilityLocation, CapabilityRegistry, ConfirmationPolicy,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceStatus {
    pub profile: String,
    pub network_enabled: bool,
    pub audio_ready: bool,
}

/// Adds the secret-free local status reader to a capability registry.
///
/// # Errors
///
/// Returns an error if the fixed descriptor conflicts with an existing tool.
pub fn register_device_status(
    registry: &mut CapabilityRegistry,
    status: DeviceStatus,
) -> Result<(), CapabilityError> {
    registry.register(
        Capability {
            name: "device_status".to_owned(),
            description: "Read Oreo's bounded local runtime status".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            location: CapabilityLocation::LocalProcess,
            risk: RiskClass::ReadOnly,
            confirmation: ConfirmationPolicy::Never,
            works_offline: true,
            timeout: Duration::from_millis(250),
            cancellable: true,
            disclosure: "Reads profile, connectivity mode, and audio readiness; no credentials or user content.".to_owned(),
        },
        Arc::new(DeviceStatusHandler(status)),
    )
}

struct DeviceStatusHandler(DeviceStatus);

impl ToolHandler for DeviceStatusHandler {
    fn call(&self, _arguments: &Value, cancellation: &CancellationToken) -> Result<ToolOutput> {
        if cancellation.is_cancelled() {
            anyhow::bail!("status request cancelled");
        }
        let structured = json!({
            "profile": self.0.profile,
            "network_enabled": self.0.network_enabled,
            "audio_ready": self.0.audio_ready,
        });
        Ok(ToolOutput {
            text: structured.to_string(),
            structured: Some(structured),
            is_error: false,
        })
    }
}

/// Adds a read-only view of explicit short- and long-term memories.
///
/// The model can retrieve user-managed records but cannot create, change, or
/// delete them. Those mutations stay behind the trusted local API.
///
/// # Errors
///
/// Returns an error if the fixed descriptor conflicts with another tool.
pub fn register_memory_recall(
    registry: &mut CapabilityRegistry,
    database_path: PathBuf,
    limits: StateLimits,
) -> Result<(), CapabilityError> {
    registry.register(
        Capability {
            name: "memory_recall".to_owned(),
            description: "Recall user-managed short- or long-term Oreo memories".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["session", "long_term"]}
                },
                "required": ["scope"],
                "additionalProperties": false
            }),
            location: CapabilityLocation::LocalProcess,
            risk: RiskClass::ReadOnly,
            confirmation: ConfirmationPolicy::Never,
            works_offline: true,
            timeout: Duration::from_millis(250),
            cancellable: true,
            disclosure:
                "Reads bounded memories explicitly managed through Oreo's trusted local interface."
                    .to_owned(),
        },
        Arc::new(MemoryRecallHandler {
            database_path,
            limits,
        }),
    )
}

struct MemoryRecallHandler {
    database_path: PathBuf,
    limits: StateLimits,
}

impl ToolHandler for MemoryRecallHandler {
    fn call(&self, arguments: &Value, cancellation: &CancellationToken) -> Result<ToolOutput> {
        if cancellation.is_cancelled() {
            anyhow::bail!("memory request cancelled");
        }
        let scope = match arguments.get("scope").and_then(Value::as_str) {
            Some("session") => MemoryScope::Session,
            Some("long_term") => MemoryScope::LongTerm,
            _ => return Ok(ToolOutput::error("scope must be session or long_term")),
        };
        let mut store = StateStore::open(&self.database_path, self.limits)?;
        let memories = store.memories(&SystemClock, scope, 32)?;
        let structured = json!({
            "scope": arguments["scope"],
            "memories": memories.into_iter().map(|memory| json!({
                "id": memory.id,
                "content": memory.content,
                "updated_at_ms": memory.updated_at_ms,
                "expires_at_ms": memory.expires_at_ms,
            })).collect::<Vec<_>>()
        });
        Ok(ToolOutput {
            text: structured.to_string(),
            structured: Some(structured),
            is_error: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crumb_agent::{AgentMode, CancellationToken};
    use serde_json::json;

    use crate::{CapabilityRegistry, DenyApprovalUi};

    use oreo_state::{MemoryScope, StateLimits, StateStore, SystemClock};

    use super::{DeviceStatus, register_device_status, register_memory_recall};

    #[test]
    fn status_tool_is_available_without_network_or_confirmation() {
        let mut registry = CapabilityRegistry::new(false, Arc::new(DenyApprovalUi));
        register_device_status(
            &mut registry,
            DeviceStatus {
                profile: "sbc".to_owned(),
                network_enabled: false,
                audio_ready: false,
            },
        )
        .expect("status capability registers");
        let (tools, approvals) = registry.finish();

        let output = tools
            .call(
                "device_status",
                &json!({}),
                AgentMode::Negotiate,
                approvals.as_ref(),
                &CancellationToken::default(),
            )
            .expect("status is permitted offline");
        assert!(output.text.contains("\"profile\":\"sbc\""));
        assert!(!output.is_error);
    }

    #[test]
    fn memory_tool_reads_only_explicit_records() {
        let root = tempfile::tempdir().expect("state root");
        let path = root.path().join("oreo.db");
        let limits = StateLimits::sbc();
        StateStore::open(&path, limits)
            .expect("state opens")
            .remember(
                &SystemClock,
                "locale",
                MemoryScope::LongTerm,
                "The user prefers British English.",
                None,
            )
            .expect("memory saves");
        let mut registry = CapabilityRegistry::new(false, Arc::new(DenyApprovalUi));
        register_memory_recall(&mut registry, path, limits).expect("memory tool registers");
        let (tools, approvals) = registry.finish();

        let output = tools
            .call(
                "memory_recall",
                &json!({"scope": "long_term"}),
                AgentMode::Negotiate,
                approvals.as_ref(),
                &CancellationToken::default(),
            )
            .expect("memory recall is allowed");
        assert!(output.text.contains("British English"));
        assert!(!output.is_error);
    }
}
