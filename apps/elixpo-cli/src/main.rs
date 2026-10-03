use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use oreo_agent::provider::{PollinationsConfig, PollinationsProvider};
use oreo_agent::tools::CancellationToken as AgentCancellation;
use oreo_agent::{
    AgentEvent, AgentProfile, CapabilityRegistry, DenyApprovalUi, DeviceStatus, EventSink,
    OreoAgent, inspect_memory, list_memory, register_device_status,
};
use oreo_audio::{
    AudioLimits, AudioOutput, AudioSource, CpalInputSource, CpalOutput, PcmChunk,
    default_audio_devices,
};
use oreo_core::{AssistantRuntime, CancellationToken, FakeHarness, RuntimeConfig, StdoutSink};
use oreo_local_api::{PROTOCOL_VERSION, Request, Response, RuntimePhase, send_request};

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
            let daemon = match send_request(
                &socket_path()?,
                &Request::Status {
                    version: PROTOCOL_VERSION,
                },
            ) {
                Ok(Response::Status {
                    schema_version,
                    active_timers,
                    ..
                }) => format!("ready (schema: {schema_version}, timers: {active_timers})"),
                _ => "offline".to_owned(),
            };
            let agent = if env::var_os("POLLINATIONS_API_KEY").is_some()
                && env::var_os("OREO_MODEL").is_some()
            {
                "configured"
            } else {
                "offline (set POLLINATIONS_API_KEY and OREO_MODEL)"
            };
            println!(
                "Oreo runtime: ready (profile: {}, queue: {}, daemon: {}, agent: {})",
                config.profile_name, config.event_capacity, daemon, agent
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
        Some("daemon") => daemon_command(arguments),
        Some("memory") => memory_command(arguments),
        Some("diagnostics") => diagnostics_command(arguments),
        Some("audio") => audio_command(arguments),
        Some("--version" | "-V") => {
            println!("elixpo {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => {
            Err(
                "usage: elixpo <status|ask [--offline]|tools|timer|daemon|memory|diagnostics|audio|--version>"
                    .into(),
            )
        }
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
            match daemon_request(&Request::TimerSet {
                version: PROTOCOL_VERSION,
                id: id.clone(),
                duration_ms,
            })? {
                Response::TimerSet { .. } => {
                    println!("Timer {id} scheduled for {seconds} seconds.");
                    Ok(())
                }
                response => unexpected_response(&response),
            }
        }
        [command] if command == "list" => {
            match daemon_request(&Request::TimerList {
                version: PROTOCOL_VERSION,
            })? {
                Response::TimerList { timers, .. } => {
                    if timers.is_empty() {
                        println!("No active timers.");
                    } else {
                        let now_ms = now_ms()?;
                        for timer in timers {
                            let remaining_seconds =
                                timer.due_at_ms.saturating_sub(now_ms).div_ceil(1_000);
                            println!("{}\t{} seconds remaining", timer.id, remaining_seconds);
                        }
                    }
                    Ok(())
                }
                response => unexpected_response(&response),
            }
        }
        [command, id] if command == "cancel" => {
            match daemon_request(&Request::TimerCancel {
                version: PROTOCOL_VERSION,
                id: id.clone(),
            })? {
                Response::TimerCancelled { .. } => {
                    println!("Timer {id} cancelled.");
                    Ok(())
                }
                response => unexpected_response(&response),
            }
        }
        _ => Err("usage: elixpo timer <set <id> <seconds>|list|cancel <id>>".into()),
    }
}

fn daemon_command(
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    match arguments.collect::<Vec<_>>().as_slice() {
        [command] if command == "stop" => match daemon_request(&Request::Shutdown {
            version: PROTOCOL_VERSION,
        })? {
            Response::ShuttingDown { .. } => {
                println!("Oreo daemon is stopping.");
                Ok(())
            }
            response => unexpected_response(&response),
        },
        _ => Err("usage: elixpo daemon stop".into()),
    }
}

fn memory_command(
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    match arguments.collect::<Vec<_>>().as_slice() {
        [command] if command == "list" => {
            let memories = list_memory(session_root()?)?;
            if memories.is_empty() {
                println!("No agent memories.");
            } else {
                for memory in memories {
                    println!(
                        "{}\tturns={}\tstatus={}\tmode={}\tarchived={}",
                        memory.id,
                        memory.turns,
                        memory.last_status.unwrap_or("none"),
                        memory.mode,
                        memory.archived
                    );
                }
            }
            Ok(())
        }
        [command, id] if command == "inspect" => {
            let memory = inspect_memory(session_root()?, id)?;
            println!("id: {}", memory.summary.id);
            println!("mode: {}", memory.summary.mode);
            println!("archived: {}", memory.summary.archived);
            println!("turns: {}", memory.summary.turns);
            println!(
                "last_status: {}",
                memory.summary.last_status.unwrap_or("none")
            );
            println!("started_at_ms: {}", memory.summary.started_at_ms);
            println!("last_event_at_ms: {}", memory.summary.last_event_at_ms);
            println!("retained_metadata_events: {}", memory.retained_events);
            Ok(())
        }
        _ => Err("usage: elixpo memory <list|inspect <id>>".into()),
    }
}

fn diagnostics_command(
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if arguments.count() != 0 {
        return Err("usage: elixpo diagnostics".into());
    }
    match daemon_request(&Request::Diagnostics {
        version: PROTOCOL_VERSION,
    })? {
        Response::Diagnostics {
            phase,
            schema_version,
            retained_events,
            event_limit,
            active_timers,
            timer_limit,
            resident_memory_kib,
            ..
        } => {
            println!("phase: {}", phase_name(phase));
            println!("state_schema: {schema_version}");
            println!("runtime_events: {retained_events}/{event_limit}");
            println!("active_timers: {active_timers}/{timer_limit}");
            match resident_memory_kib {
                Some(memory) => println!("resident_memory_kib: {memory}"),
                None => println!("resident_memory_kib: unavailable"),
            }
            Ok(())
        }
        response => unexpected_response(&response),
    }
}

const fn phase_name(phase: RuntimePhase) -> &'static str {
    match phase {
        RuntimePhase::Starting => "starting",
        RuntimePhase::Ready => "ready",
        RuntimePhase::Stopping => "stopping",
        RuntimePhase::Faulted => "faulted",
    }
}

fn audio_command(
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let arguments = arguments.collect::<Vec<_>>();
    match arguments.as_slice() {
        [command] if command == "devices" => {
            let devices = default_audio_devices()?;
            println!("host: {}", devices.host);
            println!(
                "default_input: {}",
                devices.default_input.as_deref().unwrap_or("unavailable")
            );
            println!(
                "default_output: {}",
                devices.default_output.as_deref().unwrap_or("unavailable")
            );
            Ok(())
        }
        [command, seconds] if command == "capture-test" => {
            let seconds = diagnostic_seconds(seconds)?;
            capture_test(seconds)
        }
        [command, seconds] if command == "playback-test" => {
            let seconds = diagnostic_seconds(seconds)?;
            playback_test(seconds)
        }
        _ => Err(
            "usage: elixpo audio <devices|capture-test <1-10 seconds>|playback-test <1-10 seconds>>"
                .into(),
        ),
    }
}

fn diagnostic_seconds(value: &str) -> Result<u64, Box<dyn std::error::Error>> {
    let seconds = value
        .parse::<u64>()
        .map_err(|_| "audio diagnostic duration must be 1-10 seconds")?;
    if !(1..=10).contains(&seconds) {
        return Err("audio diagnostic duration must be 1-10 seconds".into());
    }
    Ok(seconds)
}

fn capture_test(seconds: u64) -> Result<(), Box<dyn std::error::Error>> {
    let mut source = CpalInputSource::open_default(AudioLimits::sbc())?;
    let format = source.format();
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let cancellation = CancellationToken::new();
    let mut chunks = 0_u64;
    let mut samples = 0_u64;
    let mut amplitude_sum = 0_u64;
    let mut peak = 0_u16;
    while Instant::now() < deadline {
        let Some(chunk) = source.next_chunk(&cancellation)? else {
            break;
        };
        chunks = chunks.saturating_add(1);
        samples = samples.saturating_add(u64::try_from(chunk.samples().len()).unwrap_or(u64::MAX));
        for sample in chunk.samples() {
            let amplitude = sample.unsigned_abs();
            amplitude_sum = amplitude_sum.saturating_add(u64::from(amplitude));
            peak = peak.max(amplitude);
        }
    }
    source.control().stop();
    if samples == 0 {
        return Err("microphone produced no samples".into());
    }
    let stats = source.stats();
    println!(
        "format: {} Hz, {} channel(s)",
        format.sample_rate_hz, format.channels
    );
    println!("chunks: {chunks}");
    println!("mean_amplitude: {}", amplitude_sum / samples);
    println!("peak_amplitude: {peak}");
    println!("dropped_chunks: {}", stats.dropped_chunks);
    println!("stream_errors: {}", stats.stream_errors);
    if stats.dropped_chunks != 0 || stats.stream_errors != 0 {
        return Err("microphone diagnostic detected lost buffers or stream errors".into());
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn playback_test(seconds: u64) -> Result<(), Box<dyn std::error::Error>> {
    let mut output = CpalOutput::open_default(AudioLimits::sbc())?;
    let format = output.native_format();
    output.begin(format)?;
    let cancellation = CancellationToken::new();
    let frames_per_chunk = usize::try_from(format.sample_rate_hz)? / 50;
    let samples_per_chunk = frames_per_chunk
        .checked_mul(usize::from(format.channels))
        .ok_or("speaker frame is too large")?;
    let chunk_count = seconds.saturating_mul(50);
    let mut phase = 0_f32;
    let phase_step = 440_f32 * 2_f32 * std::f32::consts::PI / format.sample_rate_hz as f32;
    for _ in 0..chunk_count {
        let mut samples = Vec::with_capacity(samples_per_chunk);
        for _ in 0..frames_per_chunk {
            let value = (phase.sin() * 8_000_f32) as i16;
            phase = (phase + phase_step) % (2_f32 * std::f32::consts::PI);
            samples.extend(std::iter::repeat_n(value, usize::from(format.channels)));
        }
        output.write(&PcmChunk::new(format, samples)?, &cancellation)?;
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(100));
    let stats = output.stats();
    output.stop();
    println!(
        "format: {} Hz, {} channel(s)",
        format.sample_rate_hz, format.channels
    );
    println!("played_samples: {}", stats.played_samples);
    println!("underrun_callbacks: {}", stats.underrun_callbacks);
    println!("stream_errors: {}", stats.stream_errors);
    if stats.stream_errors != 0 {
        return Err("speaker diagnostic detected stream errors".into());
    }
    Ok(())
}

fn daemon_request(request: &Request) -> Result<Response, Box<dyn std::error::Error>> {
    let response = send_request(&socket_path()?, request)?;
    if let Response::Error { message, .. } = response {
        Err(message.into())
    } else {
        Ok(response)
    }
}

fn unexpected_response<T>(_response: &Response) -> Result<T, Box<dyn std::error::Error>> {
    Err("local daemon returned an unexpected response".into())
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

fn socket_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(state_directory()?.join("oreo.sock"))
}

fn now_ms() -> Result<u64, Box<dyn std::error::Error>> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
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
