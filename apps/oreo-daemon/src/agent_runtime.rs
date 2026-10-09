use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use oreo_agent::provider::{PollinationsConfig, PollinationsProvider};
use oreo_agent::{
    AgentEvent, AgentProfile, AgentTurnController, CapabilityRegistry, DenyApprovalUi,
    DeviceStatus, EventSink, OreoAgent, TurnSubmission, VoiceTurnPolicy, register_device_status,
};

const PERSONA: &str = include_str!("../../../config/persona.md");
const DEFAULT_MODEL: &str = "openai/gpt-5.4-nano";
const MAX_ENV_FILE_BYTES: u64 = 32 * 1_024;
const MAX_API_KEY_BYTES: usize = 4_096;
const MAX_MODEL_BYTES: usize = 256;

pub(crate) struct VoiceAgentConfig {
    api_key: String,
    model: String,
    session_root: PathBuf,
}

impl VoiceAgentConfig {
    pub(crate) fn from_environment(
        repository_root: &Path,
        session_root: PathBuf,
    ) -> Result<Self, VoiceAgentError> {
        let local = read_local_environment(repository_root)?;
        let api_key = preferred_value("POLLINATIONS_API_KEY", &local)
            .ok_or_else(|| VoiceAgentError::new("POLLINATIONS_API_KEY is unavailable"))?;
        let model =
            preferred_value("OREO_MODEL", &local).unwrap_or_else(|| DEFAULT_MODEL.to_owned());
        if api_key.len() > MAX_API_KEY_BYTES || model.len() > MAX_MODEL_BYTES {
            return Err(VoiceAgentError::new("voice agent settings are invalid"));
        }
        Ok(Self {
            api_key,
            model,
            session_root,
        })
    }
}

fn preferred_value(name: &str, local: &[(String, String)]) -> Option<String> {
    env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            local
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.clone())
                .filter(|value| !value.trim().is_empty())
        })
}

fn read_local_environment(
    repository_root: &Path,
) -> Result<Vec<(String, String)>, VoiceAgentError> {
    let path = repository_root.join(".env.local");
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(_) => return Err(VoiceAgentError::new("local agent settings are unavailable")),
    };
    if !metadata.file_type().is_file() || metadata.len() > MAX_ENV_FILE_BYTES {
        return Err(VoiceAgentError::new("local agent settings are invalid"));
    }
    let source = fs::read_to_string(path)
        .map_err(|_| VoiceAgentError::new("local agent settings are unavailable"))?;
    parse_local_environment(&source)
}

fn parse_local_environment(source: &str) -> Result<Vec<(String, String)>, VoiceAgentError> {
    let mut values = Vec::new();
    for raw_line in source.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((name, value)) = line.split_once('=') else {
            return Err(VoiceAgentError::new("local agent settings are invalid"));
        };
        let name = name.trim();
        if !matches!(name, "POLLINATIONS_API_KEY" | "OREO_MODEL") {
            continue;
        }
        if values.iter().any(|(existing, _)| existing == name) {
            return Err(VoiceAgentError::new("local agent settings are invalid"));
        }
        let value = parse_value(value.trim())?;
        values.push((name.to_owned(), value));
    }
    Ok(values)
}

fn parse_value(value: &str) -> Result<String, VoiceAgentError> {
    if value.len() >= 2 {
        let first = value.as_bytes()[0];
        let last = value.as_bytes()[value.len() - 1];
        if matches!((first, last), (b'\'', b'\'') | (b'"', b'"')) {
            return Ok(value[1..value.len() - 1].to_owned());
        }
    }
    if value.contains(['\'', '"', '$', '`', ';', '\\']) || value.contains(char::is_whitespace) {
        return Err(VoiceAgentError::new("local agent settings are invalid"));
    }
    Ok(value.to_owned())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AgentRuntimeEvent {
    Ready,
    TurnStarted,
    TurnCompleted,
    TurnCancelled,
    TurnFailed,
}

#[derive(Clone)]
pub(crate) struct VoiceAgentIngress {
    controller: Arc<Mutex<AgentTurnController>>,
    policy: Arc<VoiceTurnPolicy>,
    wake: SyncSender<()>,
    reset_requested: Arc<AtomicBool>,
}

impl VoiceAgentIngress {
    pub(crate) fn submit(&self, transcript: &str) -> Result<TurnSubmission, VoiceAgentError> {
        let submission = {
            let mut controller = self
                .controller
                .lock()
                .map_err(|_| VoiceAgentError::new("voice agent controller failed"))?;
            let action = self.policy.action(transcript, controller.is_active());
            controller
                .submit(action, transcript)
                .map_err(|_| VoiceAgentError::new("voice agent queue rejected the command"))?
        };
        signal(&self.wake)?;
        Ok(submission)
    }

    pub(crate) fn end_conversation(&self) -> Result<(), VoiceAgentError> {
        self.controller
            .lock()
            .map_err(|_| VoiceAgentError::new("voice agent controller failed"))?
            .cancel_all();
        self.reset_requested.store(true, Ordering::Release);
        signal(&self.wake)
    }
}

fn signal(sender: &SyncSender<()>) -> Result<(), VoiceAgentError> {
    match sender.try_send(()) {
        Ok(()) | Err(TrySendError::Full(())) => Ok(()),
        Err(TrySendError::Disconnected(())) => {
            Err(VoiceAgentError::new("voice agent is unavailable"))
        }
    }
}

pub(crate) struct VoiceAgentRuntime {
    ingress: VoiceAgentIngress,
    stopping: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl VoiceAgentRuntime {
    pub(crate) fn spawn(
        config: VoiceAgentConfig,
        emit: fn(AgentRuntimeEvent),
    ) -> Result<Self, VoiceAgentError> {
        let provider = Arc::new(
            PollinationsProvider::new(
                PollinationsConfig::new(config.api_key)
                    .map_err(|_| VoiceAgentError::new("Pollinations settings are invalid"))?,
            )
            .map_err(|_| VoiceAgentError::new("Pollinations provider could not start"))?,
        );
        let mut profile = AgentProfile::voice(config.model);
        PERSONA.clone_into(&mut profile.persona);
        let mut capabilities = CapabilityRegistry::new(true, Arc::new(DenyApprovalUi));
        register_device_status(
            &mut capabilities,
            DeviceStatus {
                profile: "sbc".to_owned(),
                network_enabled: true,
                audio_ready: true,
            },
        )
        .map_err(|_| VoiceAgentError::new("voice agent capabilities are invalid"))?;
        let (tools, approvals) = capabilities.finish();
        let mut agent = OreoAgent::new(
            provider,
            tools,
            approvals,
            config.session_root,
            &new_session_id()?,
            profile,
        )
        .map_err(|_| VoiceAgentError::new("voice agent session could not start"))?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .map_err(|_| VoiceAgentError::new("voice agent runtime could not start"))?;
        let controller = Arc::new(Mutex::new(
            AgentTurnController::new(8, 32 * 1_024)
                .map_err(|_| VoiceAgentError::new("voice agent queue is invalid"))?,
        ));
        let policy = Arc::new(
            VoiceTurnPolicy::embedded()
                .map_err(|_| VoiceAgentError::new("voice steering policy is invalid"))?,
        );
        let stopping = Arc::new(AtomicBool::new(false));
        let reset_requested = Arc::new(AtomicBool::new(false));
        let (wake, receiver) = mpsc::sync_channel(1);
        let ingress = VoiceAgentIngress {
            controller: controller.clone(),
            policy,
            wake,
            reset_requested: reset_requested.clone(),
        };
        let worker_stopping = stopping.clone();
        let handle = thread::Builder::new()
            .name("oreo-agent".to_owned())
            .spawn(move || {
                agent_loop(
                    &mut agent,
                    &runtime,
                    &controller,
                    &receiver,
                    &worker_stopping,
                    &reset_requested,
                    emit,
                );
            })
            .map_err(|_| VoiceAgentError::new("voice agent thread could not start"))?;
        Ok(Self {
            ingress,
            stopping,
            handle: Some(handle),
        })
    }

    pub(crate) fn ingress(&self) -> VoiceAgentIngress {
        self.ingress.clone()
    }

    pub(crate) fn shutdown(mut self) -> Result<(), VoiceAgentError> {
        self.stopping.store(true, Ordering::Release);
        self.ingress
            .controller
            .lock()
            .map_err(|_| VoiceAgentError::new("voice agent controller failed"))?
            .cancel_all();
        let _ = signal(&self.ingress.wake);
        self.handle
            .take()
            .ok_or_else(|| VoiceAgentError::new("voice agent thread is unavailable"))?
            .join()
            .map_err(|_| VoiceAgentError::new("voice agent thread stopped unexpectedly"))
    }
}

fn agent_loop(
    agent: &mut OreoAgent,
    runtime: &tokio::runtime::Runtime,
    controller: &Arc<Mutex<AgentTurnController>>,
    receiver: &Receiver<()>,
    stopping: &AtomicBool,
    reset_requested: &AtomicBool,
    emit: fn(AgentRuntimeEvent),
) {
    emit(AgentRuntimeEvent::Ready);
    while !stopping.load(Ordering::Acquire) {
        if reset_requested.swap(false, Ordering::AcqRel) {
            agent.reset_conversation();
        }
        let turn = if let Ok(mut controller) = controller.lock() {
            controller.begin_next()
        } else {
            emit(AgentRuntimeEvent::TurnFailed);
            return;
        };
        let Some(turn) = turn else {
            let _ = receiver.recv_timeout(Duration::from_millis(100));
            continue;
        };
        let mut sink = LifecycleSink { emit };
        let result = runtime.block_on(agent.ask(turn.message, &turn.cancellation, &mut sink));
        if let Ok(mut controller) = controller.lock() {
            let _ = controller.finish_active(&turn.cancellation);
        } else {
            emit(AgentRuntimeEvent::TurnFailed);
            return;
        }
        match result {
            Ok(_) => emit(AgentRuntimeEvent::TurnCompleted),
            Err(error) if error.is_cancelled() => {
                emit(AgentRuntimeEvent::TurnCancelled);
            }
            Err(_) => emit(AgentRuntimeEvent::TurnFailed),
        }
    }
}

struct LifecycleSink {
    emit: fn(AgentRuntimeEvent),
}

impl EventSink for LifecycleSink {
    fn emit(&mut self, event: AgentEvent) {
        if event == AgentEvent::Started {
            (self.emit)(AgentRuntimeEvent::TurnStarted);
        }
    }
}

fn new_session_id() -> Result<String, VoiceAgentError> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| VoiceAgentError::new("system clock is invalid"))?
        .as_millis();
    Ok(format!("oreo-daemon-{millis}-{}", std::process::id()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct VoiceAgentError {
    message: &'static str,
}

impl VoiceAgentError {
    const fn new(message: &'static str) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for VoiceAgentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for VoiceAgentError {}

#[cfg(test)]
mod tests {
    use super::{parse_local_environment, parse_value};

    #[test]
    fn local_environment_reads_only_approved_keys() {
        let values = parse_local_environment(
            "IGNORED=secret\nexport POLLINATIONS_API_KEY='test-key'\nOREO_MODEL=fixture\n",
        )
        .expect("settings parse");
        assert_eq!(
            values,
            vec![
                ("POLLINATIONS_API_KEY".to_owned(), "test-key".to_owned()),
                ("OREO_MODEL".to_owned(), "fixture".to_owned())
            ]
        );
    }

    #[test]
    fn local_environment_rejects_shell_syntax_and_duplicates() {
        assert!(parse_value("$(steal-key)").is_err());
        assert!(
            parse_local_environment("POLLINATIONS_API_KEY=one\nPOLLINATIONS_API_KEY=two").is_err()
        );
    }
}
