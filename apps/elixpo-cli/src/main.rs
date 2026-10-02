use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use oreo_agent::provider::{PollinationsConfig, PollinationsProvider};
use oreo_agent::tools::{CancellationToken as AgentCancellation, DenyAllApprovals, ToolHost};
use oreo_agent::{AgentEvent, AgentProfile, EventSink, OreoAgent};
use oreo_core::{AssistantRuntime, CancellationToken, FakeHarness, RuntimeConfig, StdoutSink};

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
            let agent = if env::var_os("POLLINATIONS_API_KEY").is_some()
                && env::var_os("OREO_MODEL").is_some()
            {
                "configured"
            } else {
                "offline (set POLLINATIONS_API_KEY and OREO_MODEL)"
            };
            println!(
                "Oreo runtime: ready (profile: {}, queue: {}, agent: {})",
                config.profile_name, config.event_capacity, agent
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
        Some("--version" | "-V") => {
            println!("elixpo {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => Err("usage: elixpo <status|ask [--offline]|--version>".into()),
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
    let mut agent = OreoAgent::new(
        provider,
        ToolHost::default(),
        Arc::new(DenyAllApprovals),
        state_root()?,
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

fn state_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = env::var_os("OREO_STATE_DIR").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path).join("sessions"));
    }
    if let Some(path) = env::var_os("XDG_STATE_HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path).join("oreo").join("sessions"));
    }
    if let Some(path) = env::var_os("HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(path)
            .join(".local")
            .join("state")
            .join("oreo")
            .join("sessions"));
    }
    Err("no state directory is available; set OREO_STATE_DIR".into())
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
