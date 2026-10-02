use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use oreo_agent::provider::{PollinationsConfig, PollinationsProvider};
use oreo_agent::tools::CancellationToken as AgentCancellation;
use oreo_agent::{
    AgentEvent, AgentProfile, CapabilityRegistry, DenyApprovalUi, DeviceStatus, EventSink,
    OreoAgent, register_device_status,
};
use oreo_core::{AssistantRuntime, CancellationToken, FakeHarness, RuntimeConfig, StdoutSink};
use oreo_state::{Clock, StateLimits, StateStore, SystemClock};

const OREO_PERSONA: &str = include_str!("../../../config/persona.md");

fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(mut arguments: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    match arguments.next().as_deref() {
        Some("status") => {
            let config = RuntimeConfig::sbc();
            config.validate()?;
            let state = state_store()?;
            let active_timers = state.scheduled_timers()?.len();
            let agent = if env::var_os("POLLINATIONS_API_KEY").is_some()
                && env::var_os("OREO_MODEL").is_some()
            {
                "configured"
            } else {
                "offline (set POLLINATIONS_API_KEY and OREO_MODEL)"
            };
            println!(
                "Oreo runtime: ready (profile: {}, queue: {}, state_schema: {}, active_timers: {}, agent: {})",
                config.profile_name,
                config.event_capacity,
                StateStore::schema_version(),
                active_timers,
                agent
            );
            Ok(())
        }
        Some("ask") => {
            let mut words = arguments.collect::<Vec<_>>();
            let offline = words.first().is_some_and(|word| word == "--offline");
            if offline {
                words.remove(0);
            }
            let request = words.join(" ");
            if request.trim().is_empty() {
                return Err("usage: elixpo ask [--offline] <request>".into());
            }
            if offline {
                offline_ask(&request)
            } else {
                live_ask(request)
            }
        }
        Some("tools") => {
            let registry = capability_registry(false)?;
            for capability in registry.capabilities() {
                println!(
                    "{}\t{:?}\t{:?}\toffline={}\t{}",
                    capability.name,
                    capability.location,
                    capability.risk,
                    capability.works_offline,
                    capability.disclosure
                );
            }
            Ok(())
        }
        Some("timer") => timer_command(arguments),
        Some("--version" | "-V") => {
            println!("elixpo {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => Err("usage: elixpo <status|ask [--offline]|tools|timer|--version>".into()),
    }
}

fn offline_ask(request: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = RuntimeConfig::sbc();
    let mut runtime = AssistantRuntime::new(config, FakeHarness, StdoutSink)?;
    runtime.handle(request, &CancellationToken::new())?;
    Ok(())
}

fn live_ask(request: String) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("POLLINATIONS_API_KEY")
        .map_err(|_| "POLLINATIONS_API_KEY is required; use --offline for the local path")?;
    let model = env::var("OREO_MODEL")
        .map_err(|_| "OREO_MODEL is required; use --offline for the local path")?;
    let provider = Arc::new(PollinationsProvider::new(PollinationsConfig::new(
        api_key,
    )?)?);
    let mut profile = AgentProfile::sbc(model);
    OREO_PERSONA.clone_into(&mut profile.persona);
    let (tools, approvals) = capability_registry(true)?.finish();
    let mut agent = OreoAgent::new(
        provider,
        tools,
        approvals,
        session_root()?,
        &new_session_id()?,
        profile,
    )?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;
    let mut output = StreamingOutput::default();
    let response =
        runtime.block_on(agent.ask(request, &AgentCancellation::default(), &mut output))?;
    if output.wrote_text {
        println!();
    } else {
        println!("{}", response.text);
    }
    Ok(())
}

fn capability_registry(
    network_enabled: bool,
) -> Result<CapabilityRegistry, Box<dyn std::error::Error>> {
    let mut registry = CapabilityRegistry::new(network_enabled, Arc::new(DenyApprovalUi));
    register_device_status(
        &mut registry,
        DeviceStatus {
            profile: "sbc".to_owned(),
            network_enabled,
            audio_ready: false,
        },
    )?;
    Ok(registry)
}

fn timer_command(
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let arguments = arguments.collect::<Vec<_>>();
    let clock = SystemClock;
    let mut state = state_store()?;
    match arguments.as_slice() {
        [command, id, seconds] if command == "set" => {
            let seconds = seconds
                .parse::<u64>()
                .map_err(|_| "timer seconds must be a positive integer")?;
            if seconds == 0 {
                return Err("timer seconds must be a positive integer".into());
            }
            let duration_ms = seconds
                .checked_mul(1_000)
                .ok_or("timer duration is too large")?;
            let due_at_ms = clock
                .now_ms()
                .checked_add(duration_ms)
                .ok_or("timer deadline is too large")?;
            state.schedule_timer(&clock, id, due_at_ms)?;
            println!("Timer {id} scheduled for {seconds} seconds.");
            Ok(())
        }
        [command] if command == "list" => {
            let timers = state.scheduled_timers()?;
            if timers.is_empty() {
                println!("No active timers.");
            } else {
                let now_ms = clock.now_ms();
                for timer in timers {
                    let remaining_seconds = timer.due_at_ms.saturating_sub(now_ms).div_ceil(1_000);
                    println!("{}\t{} seconds remaining", timer.id, remaining_seconds);
                }
            }
            Ok(())
        }
        [command, id] if command == "cancel" => {
            state.cancel_timer(id)?;
            println!("Timer {id} cancelled.");
            Ok(())
        }
        _ => Err("usage: elixpo timer <set <id> <seconds>|list|cancel <id>>".into()),
    }
}

fn state_directory() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = env::var_os("OREO_STATE_DIR").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    if let Some(path) = env::var_os("XDG_STATE_HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path).join("oreo"));
    }
    if let Some(path) = env::var_os("HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path)
            .join(".local")
            .join("state")
            .join("oreo"));
    }
    Err("no state directory is available; set OREO_STATE_DIR".into())
}

fn session_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(state_directory()?.join("sessions"))
}

fn state_store() -> Result<StateStore, Box<dyn std::error::Error>> {
    Ok(StateStore::open(
        state_directory()?.join("oreo.db"),
        StateLimits::sbc(),
    )?)
}

fn new_session_id() -> Result<String, Box<dyn std::error::Error>> {
    let millis = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    Ok(format!("oreo-{millis}-{}", std::process::id()))
}

#[derive(Default)]
struct StreamingOutput {
    wrote_text: bool,
}

impl EventSink for StreamingOutput {
    fn emit(&mut self, event: AgentEvent) {
        if let AgentEvent::TextDelta(delta) = event {
            print!("{delta}");
            let _ = io::stdout().flush();
            self.wrote_text = true;
        }
    }
}
