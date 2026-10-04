use std::fmt;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use oreo_core::CancellationToken;

use crate::{AudioError, AudioErrorKind, AudioFormat, AudioLimits, PcmChunk, StreamingSynthesizer};

const FRAME_READY: u8 = 1;
const FRAME_AUDIO: u8 = 2;
const FRAME_DONE: u8 = 3;
const TTS_FORMAT: AudioFormat = AudioFormat {
    sample_rate_hz: 24_000,
    channels: 1,
};
const MAX_FRAME_BYTES: usize = 4_800;
const POLL_INTERVAL: Duration = Duration::from_millis(20);

#[derive(Clone, Debug)]
pub struct PocketTtsConfig {
    pub python: PathBuf,
    pub worker_script: PathBuf,
    pub cache_directory: PathBuf,
    pub idle_timeout: Duration,
    pub startup_timeout: Duration,
}

impl PocketTtsConfig {
    #[must_use]
    pub fn for_repository(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self {
            python: root.join(".venv/bin/python"),
            worker_script: root.join("scripts/pocket-tts-worker.py"),
            cache_directory: root.join("models/cache/huggingface"),
            idle_timeout: Duration::from_mins(1),
            startup_timeout: Duration::from_secs(30),
        }
    }

    fn validate(&self) -> Result<(), AudioError> {
        if !self.python.is_file()
            || !self.worker_script.is_file()
            || !self.cache_directory.is_dir()
            || !(Duration::from_secs(5)..=Duration::from_hours(1)).contains(&self.idle_timeout)
            || !(Duration::from_secs(1)..=Duration::from_mins(5)).contains(&self.startup_timeout)
        {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "PocketTTS worker configuration is invalid",
            ));
        }
        Ok(())
    }
}

pub struct PocketTtsSynthesizer {
    config: PocketTtsConfig,
    max_text_bytes: usize,
    worker: Option<Worker>,
}

impl fmt::Debug for PocketTtsSynthesizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PocketTtsSynthesizer")
            .field("worker_active", &self.worker.is_some())
            .field("max_text_bytes", &self.max_text_bytes)
            .field("idle_timeout", &self.config.idle_timeout)
            .finish_non_exhaustive()
    }
}

impl PocketTtsSynthesizer {
    /// Creates a lazy local worker adapter. No model is loaded until synthesis.
    ///
    /// # Errors
    ///
    /// Rejects missing executables, scripts, cache directories, or invalid bounds.
    pub fn new(config: PocketTtsConfig, limits: AudioLimits) -> Result<Self, AudioError> {
        config.validate()?;
        let limits = limits.validate()?;
        Ok(Self {
            config,
            max_text_bytes: limits.max_response_buffer_bytes,
            worker: None,
        })
    }

    #[must_use]
    pub fn is_worker_active(&mut self) -> bool {
        if self.worker.as_mut().is_some_and(Worker::has_exited) {
            self.stop_worker();
        }
        self.worker.is_some()
    }

    pub fn shutdown(&mut self) {
        self.stop_worker();
    }

    /// Starts and warms the worker before the first speakable response chunk.
    ///
    /// # Errors
    ///
    /// Returns a redacted startup, protocol, or cancellation failure.
    pub fn prewarm(&mut self, cancellation: &CancellationToken) -> Result<(), AudioError> {
        self.start_worker(cancellation)
    }

    fn start_worker(&mut self, cancellation: &CancellationToken) -> Result<(), AudioError> {
        if self.is_worker_active() {
            return Ok(());
        }
        let mut command = Command::new(&self.config.python);
        command
            .arg("-u")
            .arg(&self.config.worker_script)
            .arg("--idle-seconds")
            .arg(self.config.idle_timeout.as_secs().to_string())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HF_HOME", &self.config.cache_directory)
            .env("HF_HUB_OFFLINE", "1")
            .env("OMP_NUM_THREADS", "2")
            .env("MKL_NUM_THREADS", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = command.spawn().map_err(|_| backend_error())?;
        let stdin = child.stdin.take().ok_or_else(backend_error)?;
        let stdout = child.stdout.take().ok_or_else(backend_error)?;
        let (sender, receiver) = sync_channel(32);
        let reader = thread::spawn(move || read_frames(stdout, &sender));
        self.worker = Some(Worker {
            child,
            stdin,
            receiver,
            reader: Some(reader),
        });

        let deadline = Instant::now() + self.config.startup_timeout;
        loop {
            if cancellation.is_cancelled() {
                self.stop_worker();
                return Err(cancelled());
            }
            if Instant::now() >= deadline {
                self.stop_worker();
                return Err(backend_error());
            }
            match self.worker_frame() {
                Ok(Some(WorkerFrame::Ready(format))) if format == TTS_FORMAT => return Ok(()),
                Ok(Some(
                    WorkerFrame::Error
                    | WorkerFrame::Done
                    | WorkerFrame::Audio(_)
                    | WorkerFrame::Ready(_),
                ))
                | Err(()) => {
                    self.stop_worker();
                    return Err(backend_error());
                }
                Ok(None) => {}
            }
        }
    }

    fn worker_frame(&self) -> Result<Option<WorkerFrame>, ()> {
        let worker = self.worker.as_ref().ok_or(())?;
        match worker.receiver.recv_timeout(POLL_INTERVAL) {
            Ok(frame) => Ok(Some(frame)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(()),
        }
    }

    fn stop_worker(&mut self) {
        if let Some(mut worker) = self.worker.take() {
            let _ = worker.child.kill();
            let _ = worker.child.wait();
            drop(worker.stdin);
            if let Some(reader) = worker.reader.take() {
                let _ = reader.join();
            }
        }
    }
}

impl StreamingSynthesizer for PocketTtsSynthesizer {
    fn synthesize(
        &mut self,
        text: &str,
        cancellation: &CancellationToken,
        emit: &mut dyn FnMut(PcmChunk) -> Result<(), AudioError>,
    ) -> Result<(), AudioError> {
        let text = text.trim();
        if text.is_empty() || text.len() > self.max_text_bytes {
            return Err(AudioError::new(
                AudioErrorKind::Capacity,
                "speech text is empty or too large",
            ));
        }
        if cancellation.is_cancelled() {
            return Err(cancelled());
        }
        self.start_worker(cancellation)?;
        let worker = self.worker.as_mut().ok_or_else(backend_error)?;
        let size = u32::try_from(text.len()).map_err(|_| backend_error())?;
        if worker.stdin.write_all(&size.to_be_bytes()).is_err()
            || worker.stdin.write_all(text.as_bytes()).is_err()
            || worker.stdin.flush().is_err()
        {
            self.stop_worker();
            return Err(backend_error());
        }

        loop {
            if cancellation.is_cancelled() {
                self.stop_worker();
                return Err(cancelled());
            }
            match self.worker_frame() {
                Ok(Some(WorkerFrame::Audio(samples))) => {
                    let chunk = PcmChunk::new(TTS_FORMAT, samples)?;
                    if let Err(error) = emit(chunk) {
                        self.stop_worker();
                        return Err(error);
                    }
                }
                Ok(Some(WorkerFrame::Done)) => return Ok(()),
                Ok(Some(WorkerFrame::Error | WorkerFrame::Ready(_))) | Err(()) => {
                    self.stop_worker();
                    return Err(backend_error());
                }
                Ok(None) => {}
            }
        }
    }
}

impl Drop for PocketTtsSynthesizer {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    receiver: Receiver<WorkerFrame>,
    reader: Option<JoinHandle<()>>,
}

impl Worker {
    fn has_exited(&mut self) -> bool {
        self.child.try_wait().is_ok_and(|status| status.is_some())
    }
}

enum WorkerFrame {
    Ready(AudioFormat),
    Audio(Vec<i16>),
    Done,
    Error,
}

fn read_frames(stdout: impl Read, sender: &std::sync::mpsc::SyncSender<WorkerFrame>) {
    let mut reader = BufReader::new(stdout);
    loop {
        let mut header = [0_u8; 5];
        if reader.read_exact(&mut header).is_err() {
            return;
        }
        let size = usize::try_from(u32::from_be_bytes([
            header[1], header[2], header[3], header[4],
        ]))
        .unwrap_or(usize::MAX);
        if size > MAX_FRAME_BYTES {
            let _ = sender.send(WorkerFrame::Error);
            return;
        }
        let mut payload = vec![0_u8; size];
        if reader.read_exact(&mut payload).is_err() {
            return;
        }
        let frame = match header[0] {
            FRAME_READY if payload.len() == 4 => WorkerFrame::Ready(AudioFormat {
                sample_rate_hz: u32::from_be_bytes([
                    payload[0], payload[1], payload[2], payload[3],
                ]),
                channels: 1,
            }),
            FRAME_AUDIO if !payload.is_empty() && payload.len().is_multiple_of(2) => {
                WorkerFrame::Audio(
                    payload
                        .chunks_exact(2)
                        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
                        .collect(),
                )
            }
            FRAME_DONE if payload.is_empty() => WorkerFrame::Done,
            _ => WorkerFrame::Error,
        };
        if sender.send(frame).is_err() {
            return;
        }
    }
}

fn backend_error() -> AudioError {
    AudioError::new(AudioErrorKind::Backend, "PocketTTS worker failed")
}

fn cancelled() -> AudioError {
    AudioError::new(AudioErrorKind::Cancelled, "speech synthesis was cancelled")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::mpsc::sync_channel;
    use std::time::Duration;

    use oreo_core::CancellationToken;

    use crate::{AudioLimits, StreamingSynthesizer};

    use super::{
        FRAME_AUDIO, FRAME_DONE, PocketTtsConfig, PocketTtsSynthesizer, WorkerFrame, read_frames,
    };

    #[test]
    fn framed_pcm_is_decoded_without_retaining_wire_bytes() {
        let mut wire = vec![FRAME_AUDIO, 0, 0, 0, 4, 1, 0, 254, 255];
        wire.extend_from_slice(&[FRAME_DONE, 0, 0, 0, 0]);
        let (sender, receiver) = sync_channel(2);
        read_frames(Cursor::new(wire), &sender);
        match receiver.recv().expect("audio frame") {
            WorkerFrame::Audio(samples) => assert_eq!(samples, [1, -2]),
            _ => panic!("expected audio frame"),
        }
        assert!(matches!(receiver.recv(), Ok(WorkerFrame::Done)));
    }

    #[test]
    fn oversized_worker_frame_is_rejected_before_allocation() {
        let wire = vec![FRAME_AUDIO, 0, 0, 18, 193];
        let (sender, receiver) = sync_channel(1);
        read_frames(Cursor::new(wire), &sender);
        assert!(matches!(receiver.recv(), Ok(WorkerFrame::Error)));
    }

    #[test]
    fn fake_worker_streams_pcm_and_is_reused() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let cache = temporary.path().join("cache");
        fs::create_dir(&cache).expect("cache directory");
        let script = temporary.path().join("fake-worker.py");
        fs::write(
            &script,
            r"import struct, sys
out = sys.stdout.buffer
source = sys.stdin.buffer
def frame(kind, payload=b''):
    out.write(bytes([kind]) + struct.pack('>I', len(payload)) + payload)
    out.flush()
frame(1, struct.pack('>I', 24000))
while True:
    header = source.read(4)
    if not header:
        break
    size = struct.unpack('>I', header)[0]
    text = source.read(size)
    if not text:
        break
    frame(2, struct.pack('<hhhh', 1, -2, 3, -4))
    frame(3)
",
        )
        .expect("fake worker script");
        let python = std::env::var_os("PYTHON")
            .map_or_else(|| PathBuf::from("/usr/bin/python3"), PathBuf::from);
        let config = PocketTtsConfig {
            python,
            worker_script: script,
            cache_directory: cache,
            idle_timeout: Duration::from_secs(5),
            startup_timeout: Duration::from_secs(2),
        };
        let mut synthesizer =
            PocketTtsSynthesizer::new(config, AudioLimits::sbc()).expect("adapter starts lazily");
        let cancellation = CancellationToken::new();
        for _ in 0..2 {
            let mut received = Vec::new();
            synthesizer
                .synthesize("Hello.", &cancellation, &mut |chunk| {
                    received.extend_from_slice(chunk.samples());
                    Ok(())
                })
                .expect("fake speech streams");
            assert_eq!(received, [1, -2, 3, -4]);
            assert!(synthesizer.is_worker_active());
        }
        synthesizer.shutdown();
        assert!(!synthesizer.is_worker_active());
    }

    #[test]
    fn pre_cancelled_request_does_not_start_worker() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let cache = temporary.path().join("cache");
        fs::create_dir(&cache).expect("cache directory");
        let script = temporary.path().join("unused.py");
        fs::write(&script, "raise SystemExit(1)\n").expect("worker script");
        let config = PocketTtsConfig {
            python: PathBuf::from("/usr/bin/python3"),
            worker_script: script,
            cache_directory: cache,
            idle_timeout: Duration::from_secs(5),
            startup_timeout: Duration::from_secs(1),
        };
        let mut synthesizer =
            PocketTtsSynthesizer::new(config, AudioLimits::sbc()).expect("adapter starts lazily");
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = synthesizer
            .synthesize("Hello.", &cancellation, &mut |_| Ok(()))
            .expect_err("cancelled request fails");
        assert_eq!(error.kind, crate::AudioErrorKind::Cancelled);
        assert!(!synthesizer.is_worker_active());
    }
}
