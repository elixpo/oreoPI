#[cfg(unix)]
fn main() -> std::process::ExitCode {
    match oreo_daemon::run_from_environment() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(unix))]
fn main() -> std::process::ExitCode {
    eprintln!("error: oreo-daemon currently requires Unix sockets");
    std::process::ExitCode::FAILURE
}
