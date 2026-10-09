//! Oreo-owned capability metadata and authorization policy.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crumb_agent::{
    ApprovalBroker, ApprovalDecision, ApprovalRequest, CancellationToken, RiskClass,
    ToolDescriptor, ToolHandler, ToolHost, ToolTransport,
};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityLocation {
    LocalProcess,
    DeviceHardware,
    CloudService,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfirmationPolicy {
    Never,
    WhenRisky,
    Always,
}

/// Complete Rust-owned policy metadata for one model-callable capability.
#[derive(Clone, Debug, PartialEq)]
pub struct Capability {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub location: CapabilityLocation,
    pub risk: RiskClass,
    pub confirmation: ConfirmationPolicy,
    pub works_offline: bool,
    pub timeout: Duration,
    pub cancellable: bool,
    pub disclosure: String,
}

impl Capability {
    fn validate(&self) -> Result<(), CapabilityError> {
        if self.name.is_empty()
            || self.name.len() > 64
            || !self.name.chars().all(|character| {
                character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
            })
        {
            return Err(CapabilityError::new("capability name is invalid"));
        }
        if self.description.trim().is_empty() || self.disclosure.trim().is_empty() {
            return Err(CapabilityError::new(
                "capability description and disclosure are required",
            ));
        }
        if !self.input_schema.is_object() {
            return Err(CapabilityError::new(
                "capability input schema must be an object",
            ));
        }
        if self.timeout.is_zero() {
            return Err(CapabilityError::new("capability timeout must be positive"));
        }
        if self.confirmation == ConfirmationPolicy::Never && self.risk != RiskClass::ReadOnly {
            return Err(CapabilityError::new(
                "mutating capabilities must require confirmation",
            ));
        }
        if matches!(
            self.risk,
            RiskClass::CredentialSensitive | RiskClass::Destructive
        ) && self.confirmation != ConfirmationPolicy::Always
        {
            return Err(CapabilityError::new(
                "sensitive capabilities must always require confirmation",
            ));
        }
        Ok(())
    }
}

/// Trusted user-interface boundary for explicit authorization.
///
/// The model receives no implementation handle and cannot call this interface.
pub trait ApprovalUi: Send + Sync {
    fn decide(
        &self,
        capability: &Capability,
        request: &ApprovalRequest,
        cancellation: &CancellationToken,
    ) -> ApprovalDecision;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DenyApprovalUi;

impl ApprovalUi for DenyApprovalUi {
    fn decide(
        &self,
        _capability: &Capability,
        _request: &ApprovalRequest,
        _cancellation: &CancellationToken,
    ) -> ApprovalDecision {
        ApprovalDecision::Deny
    }
}

/// Builds an engine tool host and its matching immutable Oreo policy together.
pub struct CapabilityRegistry {
    tools: ToolHost,
    capabilities: BTreeMap<String, Capability>,
    network_enabled: bool,
    approval_ui: Arc<dyn ApprovalUi>,
}

impl CapabilityRegistry {
    #[must_use]
    pub fn new(network_enabled: bool, approval_ui: Arc<dyn ApprovalUi>) -> Self {
        Self {
            tools: ToolHost::default(),
            capabilities: BTreeMap::new(),
            network_enabled,
            approval_ui,
        }
    }

    /// Registers metadata and its native implementation atomically.
    ///
    /// # Errors
    ///
    /// Rejects invalid metadata, duplicate names, and descriptors refused by
    /// the underlying typed tool host.
    pub fn register(
        &mut self,
        capability: Capability,
        handler: Arc<dyn ToolHandler>,
    ) -> Result<(), CapabilityError> {
        capability.validate()?;
        if self.capabilities.contains_key(&capability.name) {
            return Err(CapabilityError::new(
                "capability name is already registered",
            ));
        }
        self.tools
            .register(
                ToolDescriptor {
                    name: capability.name.clone(),
                    description: capability.description.clone(),
                    input_schema: capability.input_schema.clone(),
                    risk: capability.risk,
                    transport: ToolTransport::Native,
                },
                handler,
            )
            .map_err(|_| CapabilityError::new("capability could not be registered"))?;
        self.capabilities
            .insert(capability.name.clone(), capability);
        Ok(())
    }

    /// Registers a cloud connector under the same typed policy as native tools.
    ///
    /// # Errors
    ///
    /// Rejects non-cloud, offline, or non-network connector metadata.
    pub fn register_connector(
        &mut self,
        capability: Capability,
        handler: Arc<dyn ToolHandler>,
    ) -> Result<(), CapabilityError> {
        if capability.location != CapabilityLocation::CloudService
            || capability.works_offline
            || !matches!(
                capability.risk,
                RiskClass::NetworkAccess | RiskClass::CredentialSensitive
            )
        {
            return Err(CapabilityError::new("connector policy is invalid"));
        }
        self.register(capability, handler)
    }

    /// Registers a sensor or actuator without giving the model a hardware
    /// handle. All calls still pass through Rust-owned approval policy.
    ///
    /// # Errors
    ///
    /// Rejects metadata that is not explicitly marked as device hardware.
    pub fn register_hardware(
        &mut self,
        capability: Capability,
        handler: Arc<dyn ToolHandler>,
    ) -> Result<(), CapabilityError> {
        if capability.location != CapabilityLocation::DeviceHardware {
            return Err(CapabilityError::new("hardware policy is invalid"));
        }
        self.register(capability, handler)
    }

    pub fn capabilities(&self) -> impl Iterator<Item = &Capability> {
        self.capabilities.values()
    }

    #[must_use]
    pub fn finish(self) -> (ToolHost, Arc<dyn ApprovalBroker>) {
        let policy = CapabilityPolicy {
            capabilities: self.capabilities,
            network_enabled: self.network_enabled,
            approval_ui: self.approval_ui,
        };
        (self.tools, Arc::new(policy))
    }
}

struct CapabilityPolicy {
    capabilities: BTreeMap<String, Capability>,
    network_enabled: bool,
    approval_ui: Arc<dyn ApprovalUi>,
}

impl ApprovalBroker for CapabilityPolicy {
    fn decide(
        &self,
        request: &ApprovalRequest,
        _arguments: &Value,
        cancellation: &CancellationToken,
    ) -> ApprovalDecision {
        let Some(capability) = self.capabilities.get(&request.tool) else {
            return ApprovalDecision::Deny;
        };
        if cancellation.is_cancelled()
            || (!self.network_enabled && !capability.works_offline)
            || capability.risk != request.risk
        {
            return ApprovalDecision::Deny;
        }
        match capability.confirmation {
            ConfirmationPolicy::Never => ApprovalDecision::AllowOnce,
            ConfirmationPolicy::WhenRisky if capability.risk == RiskClass::ReadOnly => {
                ApprovalDecision::AllowOnce
            }
            ConfirmationPolicy::WhenRisky | ConfirmationPolicy::Always => {
                self.approval_ui.decide(capability, request, cancellation)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityError {
    message: &'static str,
}

impl CapabilityError {
    const fn new(message: &'static str) -> Self {
        Self { message }
    }
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl Error for CapabilityError {}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crumb_agent::{AgentMode, ToolCallErrorKind, ToolOutput};
    use serde_json::json;

    use super::*;

    struct CountingHandler(Arc<AtomicUsize>);

    impl ToolHandler for CountingHandler {
        fn call(
            &self,
            _arguments: &Value,
            _cancellation: &CancellationToken,
        ) -> anyhow::Result<ToolOutput> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(ToolOutput::text("ok"))
        }
    }

    fn capability(name: &str) -> Capability {
        Capability {
            name: name.to_owned(),
            description: "Read device status".to_owned(),
            input_schema: json!({"type":"object","additionalProperties":false}),
            location: CapabilityLocation::LocalProcess,
            risk: RiskClass::ReadOnly,
            confirmation: ConfirmationPolicy::Never,
            works_offline: true,
            timeout: Duration::from_secs(1),
            cancellable: true,
            disclosure: "Reads bounded local status metadata.".to_owned(),
        }
    }

    #[test]
    fn safe_offline_capability_runs_under_rust_policy() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = CapabilityRegistry::new(false, Arc::new(DenyApprovalUi));
        registry
            .register(
                capability("device_status"),
                Arc::new(CountingHandler(calls.clone())),
            )
            .expect("capability registers");
        let (tools, approvals) = registry.finish();

        tools
            .call(
                "device_status",
                &json!({}),
                AgentMode::Negotiate,
                approvals.as_ref(),
                &CancellationToken::default(),
            )
            .expect("read-only offline tool is allowed");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unavailable_network_capability_never_reaches_handler() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut remote = capability("remote_search");
        remote.location = CapabilityLocation::CloudService;
        remote.works_offline = false;
        let mut registry = CapabilityRegistry::new(false, Arc::new(DenyApprovalUi));
        registry
            .register(remote, Arc::new(CountingHandler(calls.clone())))
            .expect("capability registers");
        let (tools, approvals) = registry.finish();

        let error = tools
            .call(
                "remote_search",
                &json!({}),
                AgentMode::Negotiate,
                approvals.as_ref(),
                &CancellationToken::default(),
            )
            .expect_err("offline policy denies network tool");
        assert_eq!(error.kind, ToolCallErrorKind::Denied);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn sensitive_capability_requires_confirmation() {
        let mut sensitive = capability("read_credentials");
        sensitive.risk = RiskClass::CredentialSensitive;
        sensitive.confirmation = ConfirmationPolicy::WhenRisky;
        assert_eq!(
            sensitive.validate(),
            Err(CapabilityError::new(
                "sensitive capabilities must always require confirmation"
            ))
        );
    }

    #[test]
    fn connector_and_hardware_helpers_enforce_location_boundaries() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = CapabilityRegistry::new(true, Arc::new(DenyApprovalUi));
        let mut connector = capability("calendar_read");
        connector.location = CapabilityLocation::CloudService;
        connector.risk = RiskClass::NetworkAccess;
        connector.confirmation = ConfirmationPolicy::WhenRisky;
        connector.works_offline = false;
        registry
            .register_connector(connector, Arc::new(CountingHandler(calls.clone())))
            .expect("connector registers");

        let mut hardware = capability("temperature_read");
        hardware.location = CapabilityLocation::DeviceHardware;
        registry
            .register_hardware(hardware, Arc::new(CountingHandler(calls)))
            .expect("hardware registers");

        let invalid = capability("not_a_connector");
        assert!(
            registry
                .register_connector(
                    invalid,
                    Arc::new(CountingHandler(Arc::new(AtomicUsize::new(0))))
                )
                .is_err()
        );
    }
}
