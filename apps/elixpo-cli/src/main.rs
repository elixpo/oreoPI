use std::env;
use std::process::ExitCode;

use oreo_core::{AssistantRuntime, CancellationToken, FakeHarness, RuntimeConfig, StdoutSink};

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
            println!(
                "Oreo runtime: ready (profile: {}, queue: {})",
                config.profile_name, config.event_capacity
            );
            Ok(())
        }
        Some("ask") => {
            let request = arguments.collect::<Vec<_>>().join(" ");
            if request.trim().is_empty() {
                return Err("usage: elixpo ask <request>".into());
            }
            let config = RuntimeConfig::sbc();
            let mut runtime = AssistantRuntime::new(config, FakeHarness, StdoutSink)?;
            runtime.handle(&request, &CancellationToken::new())?;
            Ok(())
        }
        Some("--version" | "-V") => {
            println!("elixpo {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => Err("usage: elixpo <status|ask|--version>".into()),
    }
}
