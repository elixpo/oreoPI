use std::env;
#[cfg(feature = "vosk-stt")]
use std::fs::File;
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
#[cfg(feature = "vosk-stt")]
use oreo_audio::{
    ConvertingSource, PushToTalkState, STT_FORMAT, VoskTranscriber, WavSource, transcribe_source,
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
        Some("voice") => voice_command(arguments),
        Some("--version" | "-V") => {
            println!("elixpo {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => {
            Err(
                "usage: elixpo <status|ask [--offline]|voice [--offline]|tools|timer|daemon|memory|diagnostics|audio|--version>"
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
    let (api_key, model) = live_agent_settings()?;
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
    let runtime = agent_runtime()?;
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

fn agent_runtime() -> io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
}

fn live_agent_settings() -> Result<(String, String), Box<dyn std::error::Error>> {
    let api_key = env::var("POLLINATIONS_API_KEY")
        .map_err(|_| "POLLINATIONS_API_KEY is required; use --offline for the local path")?;
    let model = env::var("OREO_MODEL")
        .map_err(|_| "OREO_MODEL is required; use --offline for the local path")?;
    Ok((api_key, model))
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
        [command, wav_path] if command == "transcribe-test" => {
            transcribe_test(PathBuf::from(wav_path))
        }
        _ => Err(
            "usage: elixpo audio <devices|capture-test <1-10 seconds>|playback-test <1-10 seconds>|transcribe-test <wav>>"
                .into(),
        ),
    }
}

#[cfg(feature = "vosk-stt")]
fn voice_command(
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_voice_options(arguments)?;
    if !options.offline {
        let _ = live_agent_settings()?;
    }

    let limits = AudioLimits::sbc();
    let model_path = vosk_model_path();
    eprintln!("Loading the local speech model...");
    let mut transcriber = VoskTranscriber::load(model_path, limits)?;
    let mut state = PushToTalkState::default();
    state.press()?;
    let started = Instant::now();
    let cancellation = CancellationToken::new();
    let transcript_result = match options.wav_path {
        Some(wav_path) => {
            eprintln!("Using recorded voice fixture: {}", wav_path.display());
            let wav = WavSource::read(File::open(wav_path)?, limits)?;
            let mut source = ConvertingSource::new(wav, STT_FORMAT, limits)?;
            transcribe_source(&mut source, &mut transcriber, limits, &cancellation)
                .map_err(Into::into)
        }
        None => transcribe_microphone(&mut transcriber, limits, &cancellation),
    };
    let transcript = match transcript_result {
        Ok(transcript) => transcript,
        Err(error) => {
            state.fault();
            return Err(error);
        }
    };
    state.release()?;
    state.transcript_ready()?;
    println!("\nYou said: {transcript}");
    println!("utterance_ms: {}", started.elapsed().as_millis());

    let response = if options.offline {
        offline_ask(&transcript)
    } else {
        live_ask(transcript)
    };
    if let Err(error) = response {
        state.fault();
        return Err(error);
    }
    state.text_response_complete()?;
    Ok(())
}

#[cfg(feature = "vosk-stt")]
fn transcribe_microphone(
    transcriber: &mut VoskTranscriber,
    limits: AudioLimits,
    cancellation: &CancellationToken,
) -> Result<String, Box<dyn std::error::Error>> {
    wait_for_enter("Press Enter to start recording.")?;
    let input = CpalInputSource::open_default(limits)?;
    let capture_control = input.control();
    let mut source = ConvertingSource::new(input, STT_FORMAT, limits)?;

    print!("Recording... press Enter to stop. ");
    io::stdout().flush()?;
    let stopper_control = capture_control.clone();
    let stopper = std::thread::spawn(move || {
        let mut line = String::new();
        let result = io::stdin().read_line(&mut line).map(|_| ());
        stopper_control.stop();
        result
    });

    let transcript_result = transcribe_source(&mut source, transcriber, limits, cancellation);
    capture_control.stop();
    let transcript = transcript_result?;
    stopper
        .join()
        .map_err(|_| "push-to-talk input thread failed")??;
    let capture_snapshot = source.into_inner().stats();
    if capture_snapshot.dropped_chunks != 0 || capture_snapshot.stream_errors != 0 {
        return Err("microphone capture lost audio; transcript was discarded".into());
    }
    Ok(transcript)
}

#[derive(Debug, Eq, PartialEq)]
struct VoiceOptions {
    offline: bool,
    wav_path: Option<PathBuf>,
}

fn parse_voice_options(
    arguments: impl Iterator<Item = String>,
) -> Result<VoiceOptions, &'static str> {
    let mut offline = false;
    let mut wav_path = None;
    let mut arguments = arguments.peekable();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--offline" if !offline => offline = true,
            "--wav" if wav_path.is_none() => {
                let path = arguments.next().ok_or(voice_usage())?;
                if path.is_empty() {
                    return Err(voice_usage());
                }
                wav_path = Some(PathBuf::from(path));
            }
            _ => return Err(voice_usage()),
        }
    }
    Ok(VoiceOptions { offline, wav_path })
}

const fn voice_usage() -> &'static str {
    "usage: elixpo voice [--offline] [--wav <pcm-wav>]"
}

#[cfg(not(feature = "vosk-stt"))]
fn voice_command(
    _arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("voice requires a build with --features vosk-stt".into())
}

#[cfg(feature = "vosk-stt")]
fn wait_for_enter(prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    print!("{prompt} ");
    io::stdout().flush()?;
    let mut line = String::new();
    if io::stdin().read_line(&mut line)? == 0 {
        return Err("push-to-talk requires an interactive terminal".into());
    }
    Ok(())
}

#[cfg(feature = "vosk-stt")]
fn vosk_model_path() -> PathBuf {
    env::var_os("OREO_VOSK_MODEL_DIR")
        .filter(|path| !path.is_empty())
        .map_or_else(
            || PathBuf::from("models/cache/vosk-model-small-en-us-0.15"),
            PathBuf::from,
        )
}

#[cfg(feature = "vosk-stt")]
fn transcribe_test(wav_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = vosk_model_path();
    let limits = AudioLimits::sbc();
    let wav = WavSource::read(File::open(wav_path)?, limits)?;
    let mut source = ConvertingSource::new(wav, STT_FORMAT, limits)?;
    let mut transcriber = VoskTranscriber::load(model_path, limits)?;
    let cancellation = CancellationToken::new();
    let started = Instant::now();
    let transcript = transcribe_source(&mut source, &mut transcriber, limits, &cancellation)?;
    println!("transcript: {transcript}");
    println!("transcription_ms: {}", started.elapsed().as_millis());
    Ok(())
}

#[cfg(not(feature = "vosk-stt"))]
fn transcribe_test(_wav_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    Err("transcribe-test requires a build with --features vosk-stt".into())
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{VoiceOptions, agent_runtime, parse_voice_options};

    #[test]
    fn agent_runtime_supports_network_io() {
        let runtime = agent_runtime().expect("runtime builds");
        let listener = runtime
            .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
            .expect("runtime has an I/O driver");
        drop(listener);
    }

    #[test]
    fn voice_options_accept_a_repeatable_recording() {
        let options = parse_voice_options(
            ["--wav", "fixture.wav", "--offline"]
                .into_iter()
                .map(str::to_owned),
        )
        .expect("options parse");
        assert_eq!(
            options,
            VoiceOptions {
                offline: true,
                wav_path: Some(PathBuf::from("fixture.wav")),
            }
        );
    }

    #[test]
    fn voice_options_reject_duplicates_and_missing_paths() {
        for arguments in [
            vec!["--offline", "--offline"],
            vec!["--wav"],
            vec!["--wav", "one.wav", "--wav", "two.wav"],
        ] {
            assert!(
                parse_voice_options(arguments.into_iter().map(str::to_owned)).is_err(),
                "invalid options must fail"
            );
        }
    }
}
