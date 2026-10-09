//! Single-owner local daemon for Oreo state and timer scheduling.

#[cfg(all(unix, feature = "voice-agent"))]
mod agent_runtime;
#[cfg(all(unix, feature = "voice-agent"))]
mod speech_runtime;

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
    use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::thread;
    use std::time::Duration;

    #[cfg(feature = "voice-agent")]
    use crate::agent_runtime::{
        AgentRuntimeEvent, ConversationPhase, VoiceAgentConfig, VoiceAgentIngress,
        VoiceAgentRuntime,
    };
    #[cfg(feature = "voice-agent")]
    use crate::speech_runtime::SpeechRuntimeEvent;

    #[cfg(feature = "voice-runtime")]
    use oreo_audio::{
        AudioLimits, AudioSource, ConversationAudioPhase, ConvertingSource, CpalInputSource,
        OpenWakeWordConfig, OpenWakeWordDetector, STT_FORMAT, VoskTranscriber, WakeCommandPipeline,
        WakePipelineEvent,
    };
    #[cfg(feature = "voice-runtime")]
    use oreo_core::CancellationToken;
    use oreo_local_api::{
        ApiMemory, ApiMemoryScope, ApiTimer, ErrorCode, MAX_MEMORY_DURATION_MS,
        MAX_TIMER_DURATION_MS, PROTOCOL_VERSION, Request, Response, RuntimePhase, read_request,
        write_response,
    };
    use oreo_state::{
        Clock, EventOutcome, MemoryRecord, MemoryScope, RuntimeEventKind, StateError,
        StateErrorKind, StateLimits, StateStore, SystemClock,
    };

    pub struct DaemonConfig {
        pub state_directory: PathBuf,
        pub limits: StateLimits,
    }

    pub struct VoiceConfig {
        pub repository_root: PathBuf,
        #[cfg(feature = "voice-agent")]
        agent_config: Option<VoiceAgentConfig>,
        #[cfg(feature = "voice-agent")]
        agent_ingress: Option<VoiceAgentIngress>,
    }

    impl VoiceConfig {
        #[must_use]
        pub fn for_repository(root: impl AsRef<Path>) -> Self {
            Self {
                repository_root: root.as_ref().to_path_buf(),
                #[cfg(feature = "voice-agent")]
                agent_config: None,
                #[cfg(feature = "voice-agent")]
                agent_ingress: None,
            }
        }
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
        phase: AtomicU8,
        limits: StateLimits,
    }

    const PHASE_STARTING: u8 = 0;
    const PHASE_READY: u8 = 1;
    const PHASE_STOPPING: u8 = 2;
    const PHASE_FAULTED: u8 = 3;

    impl SharedState {
        fn phase(&self) -> RuntimePhase {
            match self.phase.load(Ordering::Acquire) {
                PHASE_STARTING => RuntimePhase::Starting,
                PHASE_READY => RuntimePhase::Ready,
                PHASE_STOPPING => RuntimePhase::Stopping,
                _ => RuntimePhase::Faulted,
            }
        }

        fn set_phase(&self, phase: RuntimePhase) {
            let value = match phase {
                RuntimePhase::Starting => PHASE_STARTING,
                RuntimePhase::Ready => PHASE_READY,
                RuntimePhase::Stopping => PHASE_STOPPING,
                RuntimePhase::Faulted => PHASE_FAULTED,
            };
            self.phase.store(value, Ordering::Release);
        }

        fn mark_ready(&self) {
            let _ = self.phase.compare_exchange(
                PHASE_STARTING,
                PHASE_READY,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
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
        run_inner(config, None)
    }

    fn run_inner(
        config: &DaemonConfig,
        mut voice_config: Option<VoiceConfig>,
    ) -> Result<(), DaemonError> {
        prepare_state_directory(&config.state_directory)?;
        let socket_path = config.socket_path();
        prepare_socket_path(&socket_path)?;
        let database_path = config.database_path();
        let mut store = StateStore::open(&database_path, config.limits).map_err(state_error)?;
        store
            .record_event(
                &SystemClock,
                RuntimeEventKind::RuntimeStarted,
                EventOutcome::Succeeded,
            )
            .map_err(state_error)?;
        secure_file(&database_path)?;
        let listener = UnixListener::bind(&socket_path).map_err(io_error)?;
        secure_file(&socket_path)?;
        let _socket_guard = SocketGuard(socket_path);
        let shared = Arc::new(SharedState {
            store: Mutex::new(store),
            wake: Condvar::new(),
            stopping: AtomicBool::new(false),
            phase: AtomicU8::new(PHASE_STARTING),
            limits: config.limits,
        });
        let scheduler_state = shared.clone();
        let scheduler = thread::Builder::new()
            .name("oreo-timers".to_owned())
            .spawn(move || {
                let result = scheduler_loop(&scheduler_state);
                if result.is_err() {
                    scheduler_state.set_phase(RuntimePhase::Faulted);
                    write_log(LogEvent::RuntimeFault, LogOutcome::Failed);
                }
                result
            })
            .map_err(|_| {
                DaemonError::new(
                    DaemonErrorKind::Scheduler,
                    "timer scheduler could not start",
                )
            })?;

        let agent = match spawn_voice_agent(&mut voice_config) {
            Ok(agent) => agent,
            Err(error) => {
                let _ = request_stop(&shared);
                let _ = scheduler.join();
                return Err(error);
            }
        };

        let voice = match spawn_voice(&shared, voice_config) {
            Ok(voice) => voice,
            Err(error) => {
                let _ = request_stop(&shared);
                let _ = scheduler.join();
                let _ = join_voice_agent(agent);
                return Err(error);
            }
        };

        shared.mark_ready();
        write_log(LogEvent::DaemonStarted, LogOutcome::Succeeded);

        let server_result = serve(&listener, &shared);
        if server_result.is_err() {
            shared.set_phase(RuntimePhase::Faulted);
            write_log(LogEvent::RuntimeFault, LogOutcome::Failed);
        }
        let stop_result = request_stop(&shared);
        let scheduler_result = scheduler.join().map_err(|_| {
            DaemonError::new(
                DaemonErrorKind::Scheduler,
                "timer scheduler stopped unexpectedly",
            )
        })?;
        let voice_result = join_voice(voice);
        let agent_result = join_voice_agent(agent);
        server_result
            .and(stop_result)
            .and(scheduler_result)
            .and(voice_result)
            .and(agent_result)
    }

    #[cfg(feature = "voice-agent")]
    fn spawn_voice_agent(
        voice_config: &mut Option<VoiceConfig>,
    ) -> Result<Option<VoiceAgentRuntime>, DaemonError> {
        let Some(config) = voice_config.as_mut() else {
            return Ok(None);
        };
        let Some(agent_config) = config.agent_config.take() else {
            return Ok(None);
        };
        let runtime = VoiceAgentRuntime::spawn(agent_config, agent_event, speech_event)
            .map_err(agent_error)?;
        config.agent_ingress = Some(runtime.ingress());
        Ok(Some(runtime))
    }

    #[cfg(not(feature = "voice-agent"))]
    const fn spawn_voice_agent(
        _voice_config: &mut Option<VoiceConfig>,
    ) -> Result<Option<()>, DaemonError> {
        Ok(None)
    }

    #[cfg(feature = "voice-agent")]
    fn join_voice_agent(agent: Option<VoiceAgentRuntime>) -> Result<(), DaemonError> {
        agent.map_or(Ok(()), |agent| agent.shutdown().map_err(agent_error))
    }

    #[cfg(not(feature = "voice-agent"))]
    const fn join_voice_agent(_agent: Option<()>) -> Result<(), DaemonError> {
        Ok(())
    }

    #[cfg(feature = "voice-runtime")]
    fn spawn_voice(
        shared: &Arc<SharedState>,
        config: Option<VoiceConfig>,
    ) -> Result<Option<thread::JoinHandle<()>>, DaemonError> {
        config
            .map(|voice_config| {
                let voice_state = shared.clone();
                thread::Builder::new()
                    .name("oreo-voice".to_owned())
                    .spawn(move || {
                        voice_loop(&voice_state, &voice_config);
                    })
                    .map_err(|_| {
                        DaemonError::new(DaemonErrorKind::Voice, "voice runtime could not start")
                    })
            })
            .transpose()
    }

    #[cfg(not(feature = "voice-runtime"))]
    fn spawn_voice(
        _shared: &Arc<SharedState>,
        config: Option<VoiceConfig>,
    ) -> Result<Option<()>, DaemonError> {
        if config.is_some() {
            Err(DaemonError::new(
                DaemonErrorKind::Voice,
                "voice runtime requires the voice-runtime build feature",
            ))
        } else {
            Ok(None)
        }
    }

    #[cfg(feature = "voice-runtime")]
    fn join_voice(voice: Option<thread::JoinHandle<()>>) -> Result<(), DaemonError> {
        match voice {
            Some(voice) => voice.join().map_err(|_| {
                DaemonError::new(DaemonErrorKind::Voice, "voice runtime stopped unexpectedly")
            }),
            None => Ok(()),
        }
    }

    #[cfg(not(feature = "voice-runtime"))]
    const fn join_voice(_voice: Option<()>) -> Result<(), DaemonError> {
        Ok(())
    }

    #[cfg(feature = "voice-runtime")]
    fn voice_loop(shared: &Arc<SharedState>, config: &VoiceConfig) {
        let mut retry_delay = Duration::from_secs(1);
        while !shared.stopping.load(Ordering::Acquire) {
            match voice_session(shared, config) {
                Ok(()) if shared.stopping.load(Ordering::Acquire) => return,
                Ok(()) | Err(_) => {
                    write_log(LogEvent::VoiceFault, LogOutcome::Failed);
                    let deadline = std::time::Instant::now() + retry_delay;
                    while std::time::Instant::now() < deadline {
                        if shared.stopping.load(Ordering::Acquire) {
                            return;
                        }
                        thread::sleep(Duration::from_millis(100));
                    }
                    retry_delay = retry_delay.saturating_mul(2).min(Duration::from_secs(30));
                }
            }
        }
    }

    #[cfg(feature = "voice-runtime")]
    fn voice_session(shared: &Arc<SharedState>, config: &VoiceConfig) -> Result<(), DaemonError> {
        let limits = AudioLimits::sbc();
        let cancellation = CancellationToken::new();
        let detector =
            OpenWakeWordDetector::new(OpenWakeWordConfig::for_repository(&config.repository_root))
                .map_err(voice_error)?;
        let transcriber = VoskTranscriber::load(
            config
                .repository_root
                .join("models/cache/vosk-model-small-en-us-0.15"),
            limits,
        )
        .map_err(voice_error)?;
        let mut pipeline =
            WakeCommandPipeline::new(detector, transcriber, limits).map_err(voice_error)?;
        pipeline.prewarm(&cancellation).map_err(voice_error)?;
        let input = CpalInputSource::open_default(limits).map_err(voice_error)?;
        let capture_control = input.control();
        let mut source = ConvertingSource::new(input, STT_FORMAT, limits).map_err(voice_error)?;
        write_log(LogEvent::VoiceListening, LogOutcome::Succeeded);
        #[cfg(feature = "voice-agent")]
        let mut last_phase = ConversationPhase::Listening;

        while !shared.stopping.load(Ordering::Acquire) {
            let Some(chunk) = source.next_chunk(&cancellation).map_err(voice_error)? else {
                break;
            };
            #[cfg(feature = "voice-agent")]
            update_conversation_phase(config, &mut pipeline, &mut last_phase);
            match pipeline
                .process(&chunk, &cancellation)
                .map_err(voice_error)?
            {
                Some(WakePipelineEvent::WakeAccepted { .. }) => {
                    write_log(LogEvent::WakeAccepted, LogOutcome::Succeeded);
                }
                Some(WakePipelineEvent::CommandReady { transcript }) => {
                    if transcript.trim().is_empty() {
                        return Err(DaemonError::new(
                            DaemonErrorKind::Voice,
                            "voice runtime produced an empty command",
                        ));
                    }
                    #[cfg(feature = "voice-agent")]
                    if config
                        .agent_ingress
                        .as_ref()
                        .is_some_and(|agent| agent.resembles_output(&transcript))
                    {
                        write_log(LogEvent::VoiceEchoRejected, LogOutcome::Denied);
                        continue;
                    }
                    write_log(LogEvent::VoiceCommandReady, LogOutcome::Succeeded);
                    #[cfg(feature = "voice-agent")]
                    if let Some(agent) = &config.agent_ingress {
                        match agent.submit(&transcript) {
                            Ok(oreo_agent::TurnSubmission::Queued) => {
                                write_log(LogEvent::AgentTurnQueued, LogOutcome::Succeeded);
                            }
                            Ok(oreo_agent::TurnSubmission::InterruptRequested) => {
                                write_log(LogEvent::AgentTurnInterrupted, LogOutcome::Succeeded);
                            }
                            Err(_) => {
                                write_log(LogEvent::AgentTurnRejected, LogOutcome::Denied);
                            }
                        }
                    }
                }
                Some(WakePipelineEvent::CommandTimedOut) => {
                    write_log(LogEvent::VoiceCommandTimedOut, LogOutcome::Denied);
                }
                Some(WakePipelineEvent::ConversationEnded) => {
                    write_log(LogEvent::VoiceConversationEnded, LogOutcome::Succeeded);
                    #[cfg(feature = "voice-agent")]
                    if let Some(agent) = &config.agent_ingress {
                        let _ = agent.end_conversation();
                    }
                }
                Some(WakePipelineEvent::ClarificationNeeded) => {
                    write_log(LogEvent::VoiceClarificationNeeded, LogOutcome::Denied);
                    #[cfg(feature = "voice-agent")]
                    if let Some(agent) = &config.agent_ingress {
                        let _ = agent.request_clarification();
                    }
                }
                Some(WakePipelineEvent::PossibleBargeIn { transcript }) => {
                    handle_possible_barge_in(config, &transcript);
                }
                None => {}
            }
        }
        cancellation.cancel();
        capture_control.stop();
        pipeline.shutdown();
        let snapshot = source.into_inner().stats();
        if snapshot.dropped_chunks != 0 || snapshot.stream_errors != 0 {
            return Err(DaemonError::new(
                DaemonErrorKind::Voice,
                "voice runtime lost microphone audio",
            ));
        }
        Ok(())
    }

    #[cfg(feature = "voice-agent")]
    fn handle_possible_barge_in(config: &VoiceConfig, transcript: &str) {
        if let Some(agent) = &config.agent_ingress
            && !agent.resembles_output(transcript)
            && agent
                .interrupt_speech()
                .is_ok_and(|interrupted| interrupted)
        {
            write_log(LogEvent::VoiceBargeIn, LogOutcome::Succeeded);
        }
    }

    #[cfg(not(feature = "voice-agent"))]
    fn handle_possible_barge_in(_config: &VoiceConfig, _transcript: &str) {}

    #[cfg(feature = "voice-agent")]
    fn update_conversation_phase(
        config: &VoiceConfig,
        pipeline: &mut WakeCommandPipeline,
        last_phase: &mut ConversationPhase,
    ) {
        let Some(agent) = &config.agent_ingress else {
            return;
        };
        let phase = agent.phase();
        pipeline.set_conversation_phase(match phase {
            ConversationPhase::Listening => ConversationAudioPhase::Listening,
            ConversationPhase::Thinking => ConversationAudioPhase::Thinking,
            ConversationPhase::Speaking => ConversationAudioPhase::Speaking,
        });
        if phase == *last_phase {
            return;
        }
        write_log(
            match phase {
                ConversationPhase::Listening => LogEvent::VoiceListening,
                ConversationPhase::Thinking => LogEvent::VoiceThinking,
                ConversationPhase::Speaking => LogEvent::VoiceSpeaking,
            },
            LogOutcome::Succeeded,
        );
        *last_phase = phase;
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
            write_log(LogEvent::RequestRejected, LogOutcome::Denied);
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
            write_log(LogEvent::RequestRejected, LogOutcome::Denied);
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
            Request::Diagnostics { .. } => (diagnostics_response(shared), false),
            Request::TimerSet {
                id, duration_ms, ..
            } => (set_timer_response(shared, &id, duration_ms), false),
            Request::TimerList { .. } => (timer_list_response(shared), false),
            Request::TimerCancel { id, .. } => (cancel_timer_response(shared, id), false),
            Request::MemoryRemember {
                id,
                scope,
                content,
                duration_ms,
                ..
            } => (
                remember_response(shared, &id, scope, &content, duration_ms),
                false,
            ),
            Request::MemoryList { scope, .. } => (memory_list_response(shared, scope), false),
            Request::MemoryForget { id, .. } => (forget_memory_response(shared, &id), false),
            Request::Shutdown { .. } => {
                if request_stop(shared).is_ok() {
                    write_log(LogEvent::DaemonStopping, LogOutcome::Succeeded);
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
                phase: shared.phase(),
                schema_version: StateStore::schema_version(),
                active_timers: timers.len(),
            },
            Err(_) => internal_error(),
        }
    }

    fn diagnostics_response(shared: &SharedState) -> Response {
        let snapshot = lock_store(shared).and_then(|store| {
            let retained_events = store.event_count().map_err(state_error)?;
            let active_timers = store.scheduled_timers().map_err(state_error)?.len();
            Ok((retained_events, active_timers))
        });
        match snapshot {
            Ok((retained_events, active_timers)) => Response::Diagnostics {
                version: PROTOCOL_VERSION,
                phase: shared.phase(),
                schema_version: StateStore::schema_version(),
                retained_events,
                event_limit: shared.limits.max_events,
                active_timers,
                timer_limit: shared.limits.max_active_timers,
                resident_memory_kib: resident_memory_kib(),
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
                write_log(LogEvent::TimerScheduled, LogOutcome::Succeeded);
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
                write_log(LogEvent::TimerCancelled, LogOutcome::Succeeded);
                Response::TimerCancelled {
                    version: PROTOCOL_VERSION,
                    id,
                }
            }
            Err(error) => state_response(error),
        }
    }

    fn remember_response(
        shared: &SharedState,
        id: &str,
        scope: ApiMemoryScope,
        content: &str,
        duration_ms: Option<u64>,
    ) -> Response {
        let clock = SystemClock;
        let expires_at_ms = match (scope, duration_ms) {
            (ApiMemoryScope::LongTerm, None) => None,
            (ApiMemoryScope::Session, Some(duration))
                if duration > 0 && duration <= MAX_MEMORY_DURATION_MS =>
            {
                match clock.now_ms().checked_add(duration) {
                    Some(expires) => Some(expires),
                    None => {
                        return error_response(
                            ErrorCode::InvalidRequest,
                            "memory expiry is invalid",
                        );
                    }
                }
            }
            _ => {
                return error_response(
                    ErrorCode::InvalidRequest,
                    "memory duration does not match its scope",
                );
            }
        };
        let state_scope = match scope {
            ApiMemoryScope::Session => MemoryScope::Session,
            ApiMemoryScope::LongTerm => MemoryScope::LongTerm,
        };
        match lock_store(shared).and_then(|mut store| {
            store
                .remember(&clock, id, state_scope, content, expires_at_ms)
                .map_err(state_error)
        }) {
            Ok(memory) => Response::MemoryRemembered {
                version: PROTOCOL_VERSION,
                memory: api_memory(memory),
            },
            Err(error) => state_response(error),
        }
    }

    fn memory_list_response(shared: &SharedState, scope: ApiMemoryScope) -> Response {
        let state_scope = match scope {
            ApiMemoryScope::Session => MemoryScope::Session,
            ApiMemoryScope::LongTerm => MemoryScope::LongTerm,
        };
        match lock_store(shared).and_then(|mut store| {
            store
                .memories(&SystemClock, state_scope, shared.limits.max_memories)
                .map_err(state_error)
        }) {
            Ok(memories) => Response::MemoryList {
                version: PROTOCOL_VERSION,
                memories: memories.into_iter().map(api_memory).collect(),
            },
            Err(error) => state_response(error),
        }
    }

    fn forget_memory_response(shared: &SharedState, id: &str) -> Response {
        match lock_store(shared).and_then(|mut store| store.forget_memory(id).map_err(state_error))
        {
            Ok(()) => Response::MemoryForgotten {
                version: PROTOCOL_VERSION,
                id: id.to_owned(),
            },
            Err(error) => state_response(error),
        }
    }

    fn api_memory(memory: MemoryRecord) -> ApiMemory {
        ApiMemory {
            id: memory.id,
            scope: match memory.scope {
                MemoryScope::Session => ApiMemoryScope::Session,
                MemoryScope::LongTerm => ApiMemoryScope::LongTerm,
            },
            content: memory.content,
            updated_at_ms: memory.updated_at_ms,
            expires_at_ms: memory.expires_at_ms,
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
            write_log(LogEvent::TimerFired, LogOutcome::Succeeded);
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
        if shared.phase() != RuntimePhase::Faulted {
            shared.set_phase(RuntimePhase::Stopping);
        }
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

    #[cfg(target_os = "linux")]
    fn resident_memory_kib() -> Option<u64> {
        fs::read_to_string("/proc/self/status")
            .ok()?
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmRSS:")?
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok()
            })
    }

    #[cfg(not(target_os = "linux"))]
    const fn resident_memory_kib() -> Option<u64> {
        None
    }

    #[derive(Clone, Copy)]
    enum LogEvent {
        DaemonStarted,
        DaemonStopping,
        RequestRejected,
        TimerScheduled,
        TimerCancelled,
        TimerFired,
        RuntimeFault,
        #[cfg(feature = "voice-agent")]
        AgentReady,
        #[cfg(feature = "voice-agent")]
        AgentTurnCancelled,
        #[cfg(feature = "voice-agent")]
        AgentTurnCompleted,
        #[cfg(feature = "voice-agent")]
        AgentProviderFailed,
        #[cfg(feature = "voice-agent")]
        AgentDeadlineExceeded,
        #[cfg(feature = "voice-agent")]
        AgentLimitReached,
        #[cfg(feature = "voice-agent")]
        AgentRuntimeFailed,
        #[cfg(feature = "voice-agent")]
        AgentTurnInterrupted,
        #[cfg(feature = "voice-agent")]
        AgentTurnQueued,
        #[cfg(feature = "voice-agent")]
        AgentTurnRejected,
        #[cfg(feature = "voice-agent")]
        AgentTurnStarted,
        #[cfg(feature = "voice-agent")]
        SpeechCancelled,
        #[cfg(feature = "voice-agent")]
        SpeechFailed,
        #[cfg(feature = "voice-agent")]
        SpeechFinished,
        #[cfg(feature = "voice-agent")]
        SpeechGenerating,
        #[cfg(feature = "voice-agent")]
        SpeechRuntimeReady,
        #[cfg(feature = "voice-agent")]
        SpeechStarted,
        #[cfg(feature = "voice-runtime")]
        WakeAccepted,
        #[cfg(feature = "voice-runtime")]
        VoiceCommandReady,
        #[cfg(feature = "voice-runtime")]
        VoiceCommandTimedOut,
        #[cfg(feature = "voice-runtime")]
        VoiceClarificationNeeded,
        #[cfg(feature = "voice-runtime")]
        VoiceConversationEnded,
        #[cfg(feature = "voice-agent")]
        VoiceEchoRejected,
        #[cfg(feature = "voice-agent")]
        VoiceBargeIn,
        #[cfg(feature = "voice-runtime")]
        VoiceFault,
        #[cfg(feature = "voice-runtime")]
        VoiceListening,
        #[cfg(feature = "voice-agent")]
        VoiceSpeaking,
        #[cfg(feature = "voice-agent")]
        VoiceThinking,
    }

    impl LogEvent {
        const fn as_str(self) -> &'static str {
            match self {
                Self::DaemonStarted => "daemon_started",
                Self::DaemonStopping => "daemon_stopping",
                Self::RequestRejected => "request_rejected",
                Self::TimerScheduled => "timer_scheduled",
                Self::TimerCancelled => "timer_cancelled",
                Self::TimerFired => "timer_fired",
                Self::RuntimeFault => "runtime_fault",
                #[cfg(feature = "voice-agent")]
                Self::AgentReady => "agent_ready",
                #[cfg(feature = "voice-agent")]
                Self::AgentTurnCancelled => "agent_turn_cancelled",
                #[cfg(feature = "voice-agent")]
                Self::AgentTurnCompleted => "agent_turn_completed",
                #[cfg(feature = "voice-agent")]
                Self::AgentProviderFailed => "agent_provider_failed",
                #[cfg(feature = "voice-agent")]
                Self::AgentDeadlineExceeded => "agent_deadline_exceeded",
                #[cfg(feature = "voice-agent")]
                Self::AgentLimitReached => "agent_limit_reached",
                #[cfg(feature = "voice-agent")]
                Self::AgentRuntimeFailed => "agent_runtime_failed",
                #[cfg(feature = "voice-agent")]
                Self::AgentTurnInterrupted => "agent_turn_interrupted",
                #[cfg(feature = "voice-agent")]
                Self::AgentTurnQueued => "agent_turn_queued",
                #[cfg(feature = "voice-agent")]
                Self::AgentTurnRejected => "agent_turn_rejected",
                #[cfg(feature = "voice-agent")]
                Self::AgentTurnStarted => "agent_turn_started",
                #[cfg(feature = "voice-agent")]
                Self::SpeechCancelled => "speech_cancelled",
                #[cfg(feature = "voice-agent")]
                Self::SpeechFailed => "speech_failed",
                #[cfg(feature = "voice-agent")]
                Self::SpeechFinished => "speech_finished",
                #[cfg(feature = "voice-agent")]
                Self::SpeechGenerating => "speech_generating",
                #[cfg(feature = "voice-agent")]
                Self::SpeechRuntimeReady => "speech_runtime_ready",
                #[cfg(feature = "voice-agent")]
                Self::SpeechStarted => "speech_started",
                #[cfg(feature = "voice-runtime")]
                Self::WakeAccepted => "wake_accepted",
                #[cfg(feature = "voice-runtime")]
                Self::VoiceCommandReady => "voice_command_ready",
                #[cfg(feature = "voice-runtime")]
                Self::VoiceCommandTimedOut => "voice_command_timed_out",
                #[cfg(feature = "voice-runtime")]
                Self::VoiceClarificationNeeded => "voice_clarification_needed",
                #[cfg(feature = "voice-runtime")]
                Self::VoiceConversationEnded => "voice_conversation_ended",
                #[cfg(feature = "voice-agent")]
                Self::VoiceEchoRejected => "voice_echo_rejected",
                #[cfg(feature = "voice-agent")]
                Self::VoiceBargeIn => "voice_barge_in",
                #[cfg(feature = "voice-runtime")]
                Self::VoiceFault => "voice_fault",
                #[cfg(feature = "voice-runtime")]
                Self::VoiceListening => "voice_listening",
                #[cfg(feature = "voice-agent")]
                Self::VoiceSpeaking => "voice_speaking",
                #[cfg(feature = "voice-agent")]
                Self::VoiceThinking => "voice_thinking",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum LogOutcome {
        Succeeded,
        Denied,
        Failed,
    }

    impl LogOutcome {
        const fn as_str(self) -> &'static str {
            match self {
                Self::Succeeded => "succeeded",
                Self::Denied => "denied",
                Self::Failed => "failed",
            }
        }
    }

    fn structured_log_line(event: LogEvent, outcome: LogOutcome, at_ms: u64) -> String {
        format!(
            r#"{{"at_ms":{at_ms},"component":"oreo-daemon","event":"{}","outcome":"{}"}}"#,
            event.as_str(),
            outcome.as_str()
        )
    }

    fn write_log(event: LogEvent, outcome: LogOutcome) {
        eprintln!(
            "{}",
            structured_log_line(event, outcome, SystemClock.now_ms())
        );
    }

    #[cfg(feature = "voice-agent")]
    fn agent_event(event: AgentRuntimeEvent) {
        let (event, outcome) = match event {
            AgentRuntimeEvent::Ready => (LogEvent::AgentReady, LogOutcome::Succeeded),
            AgentRuntimeEvent::TurnStarted => (LogEvent::AgentTurnStarted, LogOutcome::Succeeded),
            AgentRuntimeEvent::TurnCompleted => {
                (LogEvent::AgentTurnCompleted, LogOutcome::Succeeded)
            }
            AgentRuntimeEvent::TurnCancelled => (LogEvent::AgentTurnCancelled, LogOutcome::Denied),
            AgentRuntimeEvent::TurnFailed(kind) => (
                match kind {
                    crate::agent_runtime::AgentFailureKind::Provider => {
                        LogEvent::AgentProviderFailed
                    }
                    crate::agent_runtime::AgentFailureKind::Deadline => {
                        LogEvent::AgentDeadlineExceeded
                    }
                    crate::agent_runtime::AgentFailureKind::Limit => LogEvent::AgentLimitReached,
                    crate::agent_runtime::AgentFailureKind::Controller
                    | crate::agent_runtime::AgentFailureKind::Speech
                    | crate::agent_runtime::AgentFailureKind::Session
                    | crate::agent_runtime::AgentFailureKind::Persistence
                    | crate::agent_runtime::AgentFailureKind::Configuration => {
                        LogEvent::AgentRuntimeFailed
                    }
                },
                LogOutcome::Failed,
            ),
        };
        write_log(event, outcome);
    }

    #[cfg(feature = "voice-agent")]
    fn speech_event(event: SpeechRuntimeEvent) {
        let (event, outcome) = match event {
            SpeechRuntimeEvent::Ready => (LogEvent::SpeechRuntimeReady, LogOutcome::Succeeded),
            SpeechRuntimeEvent::Generating => (LogEvent::SpeechGenerating, LogOutcome::Succeeded),
            SpeechRuntimeEvent::Started => (LogEvent::SpeechStarted, LogOutcome::Succeeded),
            SpeechRuntimeEvent::Finished => (LogEvent::SpeechFinished, LogOutcome::Succeeded),
            SpeechRuntimeEvent::Cancelled => (LogEvent::SpeechCancelled, LogOutcome::Denied),
            SpeechRuntimeEvent::Failed => (LogEvent::SpeechFailed, LogOutcome::Failed),
        };
        write_log(event, outcome);
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
        Voice,
        Agent,
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
        let config = DaemonConfig {
            state_directory: state_directory()?,
            limits: StateLimits::sbc(),
        };
        if voice_enabled()? {
            #[cfg(feature = "voice-runtime")]
            {
                let root = env::var_os("OREO_REPO_ROOT")
                    .filter(|path| !path.is_empty())
                    .map_or_else(
                        || env::current_dir().map_err(io_error),
                        |path| Ok(PathBuf::from(path)),
                    )?;
                let voice = configure_voice_agent(
                    VoiceConfig::for_repository(&root),
                    &root,
                    &config.state_directory,
                )?;
                return run_inner(&config, Some(voice));
            }
            #[cfg(not(feature = "voice-runtime"))]
            {
                return Err(DaemonError::new(
                    DaemonErrorKind::Voice,
                    "voice runtime requires the voice-runtime build feature",
                ));
            }
        }
        run(&config)
    }

    fn voice_enabled() -> Result<bool, DaemonError> {
        match env::var("OREO_VOICE_ENABLED") {
            Err(env::VarError::NotPresent) => Ok(false),
            Ok(value) if matches!(value.as_str(), "1" | "true") => Ok(true),
            Ok(value) if matches!(value.as_str(), "0" | "false") => Ok(false),
            Ok(_) | Err(env::VarError::NotUnicode(_)) => Err(DaemonError::new(
                DaemonErrorKind::Voice,
                "OREO_VOICE_ENABLED must be true, false, 1, or 0",
            )),
        }
    }

    #[cfg(feature = "voice-runtime")]
    fn agent_enabled() -> Result<bool, DaemonError> {
        match env::var("OREO_AGENT_ENABLED") {
            Err(env::VarError::NotPresent) => Ok(false),
            Ok(value) if matches!(value.as_str(), "1" | "true") => Ok(true),
            Ok(value) if matches!(value.as_str(), "0" | "false") => Ok(false),
            Ok(_) | Err(env::VarError::NotUnicode(_)) => Err(DaemonError::new(
                DaemonErrorKind::Agent,
                "OREO_AGENT_ENABLED must be true, false, 1, or 0",
            )),
        }
    }

    #[cfg(feature = "voice-agent")]
    fn configure_voice_agent(
        mut voice: VoiceConfig,
        repository_root: &Path,
        state_directory: &Path,
    ) -> Result<VoiceConfig, DaemonError> {
        if agent_enabled()? {
            voice.agent_config = Some(
                VoiceAgentConfig::from_environment(
                    repository_root,
                    state_directory.join("sessions"),
                )
                .map_err(agent_error)?,
            );
        }
        Ok(voice)
    }

    #[cfg(all(feature = "voice-runtime", not(feature = "voice-agent")))]
    fn configure_voice_agent(
        voice: VoiceConfig,
        _repository_root: &Path,
        _state_directory: &Path,
    ) -> Result<VoiceConfig, DaemonError> {
        if agent_enabled()? {
            return Err(DaemonError::new(
                DaemonErrorKind::Agent,
                "voice agent requires the voice-agent build feature",
            ));
        }
        Ok(voice)
    }

    #[cfg(feature = "voice-runtime")]
    fn voice_error(_error: oreo_audio::AudioError) -> DaemonError {
        DaemonError::new(DaemonErrorKind::Voice, "voice runtime failed")
    }

    #[cfg(feature = "voice-agent")]
    fn agent_error(_error: crate::agent_runtime::VoiceAgentError) -> DaemonError {
        DaemonError::new(DaemonErrorKind::Agent, "voice agent failed")
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

        use oreo_local_api::{
            ApiMemoryScope, PROTOCOL_VERSION, Request, Response, RuntimePhase, send_request,
        };
        use oreo_state::{ManualClock, StateLimits, StateStore};

        use super::{DaemonConfig, LogEvent, LogOutcome, process_due, run, structured_log_line};

        fn limits() -> StateLimits {
            StateLimits {
                max_events: 16,
                max_active_timers: 8,
                max_timer_history: 8,
                max_memories: 16,
                max_memory_bytes: 256,
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
        fn structured_logs_have_only_fixed_redacted_fields() {
            let line = structured_log_line(LogEvent::TimerFired, LogOutcome::Succeeded, 42);
            let value: serde_json::Value = serde_json::from_str(&line).expect("log is JSON");
            assert_eq!(value["at_ms"], 42);
            assert_eq!(value["component"], "oreo-daemon");
            assert_eq!(value["event"], "timer_fired");
            assert_eq!(value["outcome"], "succeeded");
            assert_eq!(value.as_object().expect("log is an object").len(), 4);
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
            let remembered = send_request(
                &socket,
                &Request::MemoryRemember {
                    version: PROTOCOL_VERSION,
                    id: "tea-preference".to_owned(),
                    scope: ApiMemoryScope::LongTerm,
                    content: "The user prefers ginger tea.".to_owned(),
                    duration_ms: None,
                },
            )
            .expect("memory request succeeds");
            assert!(matches!(remembered, Response::MemoryRemembered { .. }));
            let memories = send_request(
                &socket,
                &Request::MemoryList {
                    version: PROTOCOL_VERSION,
                    scope: ApiMemoryScope::LongTerm,
                },
            )
            .expect("memory list succeeds");
            assert!(
                matches!(memories, Response::MemoryList { ref memories, .. } if memories.len() == 1)
            );
            let diagnostics = send_request(
                &socket,
                &Request::Diagnostics {
                    version: PROTOCOL_VERSION,
                },
            )
            .expect("diagnostics succeeds");
            assert!(matches!(
                diagnostics,
                Response::Diagnostics {
                    phase: RuntimePhase::Ready,
                    retained_events: 1,
                    active_timers: 0,
                    ..
                }
            ));
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
pub use unix::{
    DaemonConfig, DaemonError, DaemonErrorKind, VoiceConfig, run, run_from_environment,
};
