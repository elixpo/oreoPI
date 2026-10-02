use std::error::Error;
use std::fmt;

/// Non-secret runtime limits. Credentials belong in an OS credential store.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub profile_name: String,
    pub event_capacity: usize,
    pub max_input_bytes: usize,
    pub max_response_bytes: usize,
    pub idle_memory_target_mb: u16,
    pub memory_hard_limit_mb: u16,
}

impl RuntimeConfig {
    #[must_use]
    pub fn sbc() -> Self {
        Self {
            profile_name: "sbc".to_owned(),
            event_capacity: 64,
            max_input_bytes: 16 * 1024,
            max_response_bytes: 64 * 1024,
            idle_memory_target_mb: 700,
            memory_hard_limit_mb: 1_200,
        }
    }

    /// Rejects unusable or internally inconsistent bounds.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] when a name is blank, a bound is zero, or the
    /// target memory exceeds the hard ceiling.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.profile_name.trim().is_empty() {
            return Err(ConfigError("profile name cannot be empty"));
        }
        if self.event_capacity == 0
            || self.max_input_bytes == 0
            || self.max_response_bytes == 0
            || self.idle_memory_target_mb == 0
            || self.memory_hard_limit_mb == 0
        {
            return Err(ConfigError("runtime limits must be positive"));
        }
        if self.idle_memory_target_mb > self.memory_hard_limit_mb {
            return Err(ConfigError(
                "idle memory target cannot exceed the hard limit",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfigError(pub(crate) &'static str);

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::RuntimeConfig;

    #[test]
    fn sbc_profile_is_valid_and_bounded() {
        let config = RuntimeConfig::sbc();
        assert!(config.validate().is_ok());
        assert!(config.idle_memory_target_mb <= 700);
        assert!(config.memory_hard_limit_mb <= 1_200);
    }

    #[test]
    fn zero_queue_capacity_is_rejected() {
        let mut config = RuntimeConfig::sbc();
        config.event_capacity = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn configuration_contains_no_credentials() {
        let debug = format!("{:?}", RuntimeConfig::sbc()).to_ascii_lowercase();
        for forbidden in ["password", "api_key", "token", "secret"] {
            assert!(!debug.contains(forbidden));
        }
    }
}

