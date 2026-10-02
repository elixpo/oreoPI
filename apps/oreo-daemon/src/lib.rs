//! Single-owner local daemon for Oreo state and timer scheduling.

#[cfg(unix)]
mod unix {
    use std::env;
    use std::error::Error;
    use std::fmt;
    use std::fs;
    use std::io;
    use std::os::unix::fs::{FileTypeExt, PermissionsExt};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::thread;
    use std::time::Duration;

    use oreo_local_api::{
        ApiTimer, ErrorCode, MAX_TIMER_DURATION_MS, PROTOCOL_VERSION, Request, Response,
        read_request, write_response,
    };
    use oreo_state::{
        Clock, EventOutcome, RuntimeEventKind, StateError, StateErrorKind, StateLimits, StateStore,
        SystemClock,
    };

    pub struct DaemonConfig {
        pub state_directory: PathBuf,
        pub limits: StateLimits,
    }

    impl DaemonConfig {
        #[must_use]
        pub fn socket_path(&self) -> PathBuf {
            self.state_directory.join("oreo.sock")
        }

        #[must_use]
        pub fn database_path(&self) -> PathBuf {
            self.state_directory.join("oreo.db")
        }
    }

    struct SharedState {
        store: Mutex<StateStore>,
        wake: Condvar,
        stopping: AtomicBool,
    }

    struct SocketGuard(PathBuf);

    impl Drop for SocketGuard {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    /// Starts the daemon using the configured Oreo state directory.
    ///
    /// # Errors
    ///
    /// Returns a redacted error if the state path, database, socket, protocol,
    /// or scheduler cannot be initialized.
    pub fn run(config: &DaemonConfig) -> Result<(), DaemonError> {
        prepare_state_directory(&config.state_directory)?;
        let socket_path = config.socket_path();
        prepare_socket_path(&socket_path)?;
        let database_path = config.database_path();
        let store = StateStore::open(&database_path, config.limits).map_err(state_error)?;
        secure_file(&database_path)?;
        let listener = UnixListener::bind(&socket_path).map_err(io_error)?;
        secure_file(&socket_path)?;
        let _socket_guard = SocketGuard(socket_path);
        let shared = Arc::new(SharedState {
            store: Mutex::new(store),
            wake: Condvar::new(),
            stopping: AtomicBool::new(false),
        });
        let scheduler_state = shared.clone();
        let scheduler = thread::Builder::new()
            .name("oreo-timers".to_owned())
            .spawn(move || scheduler_loop(&scheduler_state))
            .map_err(|_| {
                DaemonError::new(
                    DaemonErrorKind::Scheduler,
                    "timer scheduler could not start",
                )
            })?;

        let server_result = serve(&listener, &shared);
        let stop_result = request_stop(&shared);
        let scheduler_result = scheduler.join().map_err(|_| {
            DaemonError::new(
                DaemonErrorKind::Scheduler,
                "timer scheduler stopped unexpectedly",
            )
        })?;
        server_result.and(stop_result).and(scheduler_result)
    }

    fn serve(listener: &UnixListener, shared: &Arc<SharedState>) -> Result<(), DaemonError> {
        loop {
            let (mut stream, _) = listener.accept().map_err(io_error)?;
            if handle_connection(&mut stream, shared)? {
                return Ok(());
            }
        }
    }

    fn handle_connection(
        stream: &mut UnixStream,
        shared: &Arc<SharedState>,
    ) -> Result<bool, DaemonError> {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .and_then(|()| stream.set_write_timeout(Some(Duration::from_secs(2))))
            .map_err(io_error)?;
        let Ok(request) = read_request(stream) else {
            write_response(
                stream,
                &error_response(ErrorCode::InvalidRequest, "request is invalid"),
            )
            .map_err(protocol_error)?;
            return Ok(false);
        };
        let (response, shutdown) = handle_request(shared, request);
        write_response(stream, &response).map_err(protocol_error)?;
        Ok(shutdown)
    }

    fn handle_request(shared: &Arc<SharedState>, request: Request) -> (Response, bool) {
        if request.version() != PROTOCOL_VERSION {
            return (
                error_response(
                    ErrorCode::UnsupportedVersion,
                    "protocol version is unsupported",
                ),
                false,
            );
        }
        match request {
            Request::Status { .. } => (status_response(shared), false),
            Request::TimerSet {
                id, duration_ms, ..
            } => (set_timer_response(shared, &id, duration_ms), false),
            Request::TimerList { .. } => (timer_list_response(shared), false),
            Request::TimerCancel { id, .. } => (cancel_timer_response(shared, id), false),
            Request::Shutdown { .. } => {
                if request_stop(shared).is_ok() {
                    (
                        Response::ShuttingDown {
                            version: PROTOCOL_VERSION,
                        },
                        true,
                    )
                } else {
                    (internal_error(), false)
                }
            }
        }
    }

    fn status_response(shared: &SharedState) -> Response {
        match lock_store(shared).and_then(|store| store.scheduled_timers().map_err(state_error)) {
            Ok(timers) => Response::Status {
                version: PROTOCOL_VERSION,
                schema_version: StateStore::schema_version(),
                active_timers: timers.len(),
            },
            Err(_) => internal_error(),
        }
    }

    fn set_timer_response(shared: &SharedState, id: &str, duration_ms: u64) -> Response {
        if duration_ms == 0 || duration_ms > MAX_TIMER_DURATION_MS {
            return error_response(
                ErrorCode::InvalidRequest,
                "timer duration is outside the supported range",
            );
        }
        let clock = SystemClock;
        let Some(due_at_ms) = clock.now_ms().checked_add(duration_ms) else {
            return error_response(ErrorCode::InvalidRequest, "timer deadline is invalid");
        };
        match lock_store(shared).and_then(|mut store| {
            store
                .schedule_timer(&clock, id, due_at_ms)
                .map_err(state_error)
        }) {
            Ok(timer) => {
                shared.wake.notify_all();
                Response::TimerSet {
                    version: PROTOCOL_VERSION,
                    timer: ApiTimer {
                        id: timer.id,
                        due_at_ms: timer.due_at_ms,
                        created_at_ms: timer.created_at_ms,
                    },
                }
            }
            Err(error) => state_response(error),
        }
    }

    fn timer_list_response(shared: &SharedState) -> Response {
        match lock_store(shared).and_then(|store| store.scheduled_timers().map_err(state_error)) {
            Ok(timers) => Response::TimerList {
                version: PROTOCOL_VERSION,
                timers: timers
                    .into_iter()
                    .map(|timer| ApiTimer {
                        id: timer.id,
                        due_at_ms: timer.due_at_ms,
                        created_at_ms: timer.created_at_ms,
                    })
                    .collect(),
            },
            Err(_) => internal_error(),
        }
    }

    fn cancel_timer_response(shared: &SharedState, id: String) -> Response {
        match lock_store(shared).and_then(|mut store| store.cancel_timer(&id).map_err(state_error))
        {
            Ok(()) => {
                shared.wake.notify_all();
                Response::TimerCancelled {
                    version: PROTOCOL_VERSION,
                    id,
                }
            }
            Err(error) => state_response(error),
        }
    }

    fn scheduler_loop(shared: &Arc<SharedState>) -> Result<(), DaemonError> {
        let clock = SystemClock;
        let mut store = lock_store(shared)?;
        loop {
            if shared.stopping.load(Ordering::Acquire) {
                return Ok(());
            }
            process_due(&mut store, &clock)?;
            let wait = store
                .scheduled_timers()
                .map_err(state_error)?
                .first()
                .map(|timer| Duration::from_millis(timer.due_at_ms.saturating_sub(clock.now_ms())));
            store = match wait {
                Some(duration) if duration.is_zero() => continue,
                Some(duration) => {
                    shared
                        .wake
                        .wait_timeout(store, duration)
                        .map_err(lock_error)?
                        .0
                }
                None => shared.wake.wait(store).map_err(lock_error)?,
            };
        }
    }

    fn process_due(store: &mut StateStore, clock: &dyn Clock) -> Result<Vec<String>, DaemonError> {
        let timers = store.due_timers(clock).map_err(state_error)?;
        let mut fired = Vec::with_capacity(timers.len());
        for timer in timers {
            store.mark_timer_fired(&timer.id).map_err(state_error)?;
            store
                .record_event(clock, RuntimeEventKind::TimerFired, EventOutcome::Succeeded)
                .map_err(state_error)?;
            fired.push(timer.id);
        }
        Ok(fired)
    }

    fn lock_store(shared: &SharedState) -> Result<MutexGuard<'_, StateStore>, DaemonError> {
        shared.store.lock().map_err(lock_error)
    }

    fn request_stop(shared: &SharedState) -> Result<(), DaemonError> {
        let guard = lock_store(shared)?;
        shared.stopping.store(true, Ordering::Release);
        drop(guard);
        shared.wake.notify_all();
        Ok(())
    }

    fn prepare_state_directory(path: &Path) -> Result<(), DaemonError> {
        fs::create_dir_all(path).map_err(io_error)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(io_error)
    }

    fn prepare_socket_path(path: &Path) -> Result<(), DaemonError> {
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(io_error(error)),
        };
        if !metadata.file_type().is_socket() {
            return Err(DaemonError::new(
                DaemonErrorKind::Socket,
                "local API path exists and is not a socket",
            ));
        }
        if UnixStream::connect(path).is_ok() {
            return Err(DaemonError::new(
                DaemonErrorKind::AlreadyRunning,
                "another Oreo daemon is already running",
            ));
        }
        fs::remove_file(path).map_err(io_error)
    }

    fn secure_file(path: &Path) -> Result<(), DaemonError> {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(io_error)
    }

    fn error_response(code: ErrorCode, message: &str) -> Response {
        Response::Error {
            version: PROTOCOL_VERSION,
            code,
            message: message.to_owned(),
        }
    }

    fn internal_error() -> Response {
        error_response(ErrorCode::Internal, "local daemon operation failed")
    }

    fn state_response(error: DaemonError) -> Response {
        let code = match error.kind {
            DaemonErrorKind::State(StateErrorKind::InvalidInput) => ErrorCode::InvalidRequest,
            DaemonErrorKind::State(StateErrorKind::AlreadyExists) => ErrorCode::AlreadyExists,
            DaemonErrorKind::State(StateErrorKind::NotFound) => ErrorCode::NotFound,
            DaemonErrorKind::State(StateErrorKind::Capacity) => ErrorCode::Capacity,
            _ => ErrorCode::Internal,
        };
        let message = match code {
            ErrorCode::InvalidRequest => "request is invalid",
            ErrorCode::AlreadyExists => "resource already exists",
            ErrorCode::NotFound => "resource was not found",
            ErrorCode::Capacity => "runtime capacity reached",
            _ => "local daemon operation failed",
        };
        error_response(code, message)
    }

    fn state_error(error: StateError) -> DaemonError {
        DaemonError::new(
            DaemonErrorKind::State(error.kind),
            "local state operation failed",
        )
    }

    fn io_error(_error: io::Error) -> DaemonError {
        DaemonError::new(DaemonErrorKind::Socket, "local daemon I/O failed")
    }

    fn protocol_error(_error: oreo_local_api::LocalApiError) -> DaemonError {
        DaemonError::new(DaemonErrorKind::Protocol, "local API protocol failed")
    }

    fn lock_error<T>(_error: std::sync::PoisonError<T>) -> DaemonError {
        DaemonError::new(DaemonErrorKind::StatePoisoned, "local state lock failed")
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum DaemonErrorKind {
        AlreadyRunning,
        Socket,
        Protocol,
        Scheduler,
        StatePoisoned,
        State(StateErrorKind),
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct DaemonError {
        pub kind: DaemonErrorKind,
        message: &'static str,
    }

    impl DaemonError {
        const fn new(kind: DaemonErrorKind, message: &'static str) -> Self {
            Self { kind, message }
        }
    }

    impl fmt::Display for DaemonError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.message)
        }
    }

    impl Error for DaemonError {}

    /// Resolves the default state directory and runs until a shutdown request.
    ///
    /// # Errors
    ///
    /// Returns a redacted configuration or daemon failure.
    pub fn run_from_environment() -> Result<(), DaemonError> {
        run(&DaemonConfig {
            state_directory: state_directory()?,
            limits: StateLimits::sbc(),
        })
    }

    fn state_directory() -> Result<PathBuf, DaemonError> {
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
        Err(DaemonError::new(
            DaemonErrorKind::Socket,
            "no state directory is available; set OREO_STATE_DIR",
        ))
    }

    #[cfg(test)]
    mod tests {
        use std::thread;
        use std::time::Duration;

        use oreo_local_api::{PROTOCOL_VERSION, Request, Response, send_request};
        use oreo_state::{ManualClock, StateLimits, StateStore};

        use super::{DaemonConfig, process_due, run};

        fn limits() -> StateLimits {
            StateLimits {
                max_events: 16,
                max_active_timers: 8,
                max_timer_history: 8,
            }
        }

        #[test]
        fn manual_clock_fires_due_timers_without_polling() {
            let mut store = StateStore::in_memory(limits()).expect("state opens");
            let clock = ManualClock::new(1_000);
            store
                .schedule_timer(&clock, "tea", 2_000)
                .expect("timer schedules");
            assert!(
                process_due(&mut store, &clock)
                    .expect("scheduler runs")
                    .is_empty()
            );
            clock.set(2_000);
            assert_eq!(
                process_due(&mut store, &clock).expect("scheduler runs"),
                vec!["tea"]
            );
            assert!(store.scheduled_timers().expect("timers list").is_empty());
        }

        #[test]
        fn socket_supports_reconnect_timer_flow_and_shutdown() {
            let root = tempfile::tempdir().expect("state root");
            let config = DaemonConfig {
                state_directory: root.path().to_path_buf(),
                limits: limits(),
            };
            let socket = config.socket_path();
            let server = thread::spawn(move || run(&config));
            for _ in 0..100 {
                if socket.exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            if !socket.exists() {
                if server.is_finished() {
                    let result = server.join().expect("server thread joins");
                    panic!("daemon exited before creating its socket: {result:?}");
                }
                panic!("daemon did not create its socket before the deadline");
            }

            let set = send_request(
                &socket,
                &Request::TimerSet {
                    version: PROTOCOL_VERSION,
                    id: "tea".to_owned(),
                    duration_ms: 60_000,
                },
            )
            .expect("timer request succeeds");
            assert!(matches!(set, Response::TimerSet { .. }));
            let list = send_request(
                &socket,
                &Request::TimerList {
                    version: PROTOCOL_VERSION,
                },
            )
            .expect("reconnected list succeeds");
            assert!(matches!(list, Response::TimerList { ref timers, .. } if timers.len() == 1));
            let cancel = send_request(
                &socket,
                &Request::TimerCancel {
                    version: PROTOCOL_VERSION,
                    id: "tea".to_owned(),
                },
            )
            .expect("cancel succeeds");
            assert!(matches!(cancel, Response::TimerCancelled { .. }));
            let shutdown = send_request(
                &socket,
                &Request::Shutdown {
                    version: PROTOCOL_VERSION,
                },
            )
            .expect("shutdown succeeds");
            assert!(matches!(shutdown, Response::ShuttingDown { .. }));
            server
                .join()
                .expect("server thread joins")
                .expect("daemon exits cleanly");
            assert!(!socket.exists());
        }
    }
}

#[cfg(unix)]
pub use unix::{DaemonConfig, DaemonError, DaemonErrorKind, run, run_from_environment};
