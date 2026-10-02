//! Small, deterministic capabilities available on laptops and SBCs.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crumb_agent::{CancellationToken, RiskClass, ToolHandler, ToolOutput};
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crumb_agent::{AgentMode, CancellationToken};
    use serde_json::json;

    use crate::{CapabilityRegistry, DenyApprovalUi};

    use super::{DeviceStatus, register_device_status};

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
}
