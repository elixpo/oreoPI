use std::fmt;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use oreo_core::CancellationToken;

use crate::{AudioError, AudioErrorKind, PcmChunk, STT_FORMAT};

const REQUEST_AUDIO: u8 = 1;
const REQUEST_RESET: u8 = 2;
const FRAME_READY: u8 = 1;
const FRAME_SCORE: u8 = 2;
const FRAME_RESET: u8 = 3;
const CHUNK_FRAMES: usize = 1_280;
const CHUNK_BYTES: usize = CHUNK_FRAMES * 2;
const POLL_INTERVAL: Duration = Duration::from_millis(20);

#[derive(Clone, Debug)]
pub struct OpenWakeWordConfig {
    pub python: PathBuf,
    pub worker_script: PathBuf,
    pub model: PathBuf,
    pub melspectrogram: PathBuf,
    pub embedding: PathBuf,
    pub threshold: f32,
    pub startup_timeout: Duration,
}

impl OpenWakeWordConfig {
    #[must_use]
    pub fn for_repository(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        let features = root.join("models/cache/openwakeword-v0.5.1-features");
        Self {
            python: root.join(".venv-wake/bin/python"),
            worker_script: root.join("scripts/openwakeword-worker.py"),
            model: root.join("models/cache/openwakeword-oreo/oreo.onnx"),
            melspectrogram: features.join("melspectrogram.onnx"),
            embedding: features.join("embedding_model.onnx"),
            threshold: 0.005,
            startup_timeout: Duration::from_secs(30),
        }
    }

    fn validate(&self) -> Result<(), AudioError> {
        if !self.python.is_file()
            || !self.worker_script.is_file()
            || !self.model.is_file()
            || !self.melspectrogram.is_file()
            || !self.embedding.is_file()
            || !(0.0..=1.0).contains(&self.threshold)
            || self.threshold == 0.0
            || !(Duration::from_secs(1)..=Duration::from_mins(5)).contains(&self.startup_timeout)
        {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "openWakeWord worker configuration is invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WakeCandidate {
    pub score: f32,
}

pub struct OpenWakeWordDetector {
    config: OpenWakeWordConfig,
    pending: Vec<i16>,
    worker: Option<Worker>,
}

impl fmt::Debug for OpenWakeWordDetector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenWakeWordDetector")
            .field("worker_active", &self.worker.is_some())
            .field("pending_samples", &self.pending.len())
            .field("threshold", &self.config.threshold)
            .finish_non_exhaustive()
    }
}

impl OpenWakeWordDetector {
    /// Creates a lazy detector backed by the pinned local Python worker.
    ///
    /// # Errors
    ///
    /// Rejects missing worker/model files and unsafe thresholds or timeouts.
    pub fn new(config: OpenWakeWordConfig) -> Result<Self, AudioError> {
        config.validate()?;
        Ok(Self {
            config,
            pending: Vec::with_capacity(CHUNK_FRAMES),
            worker: None,
        })
    }

    /// Starts and warms the worker without consuming microphone audio.
    ///
    /// # Errors
    ///
    /// Returns a redacted startup, protocol, or cancellation error.
    pub fn prewarm(&mut self, cancellation: &CancellationToken) -> Result<(), AudioError> {
        self.start_worker(cancellation)
    }

    /// Consumes one STT-format chunk and returns a permissive acoustic candidate.
    ///
    /// # Errors
    ///
    /// Rejects format changes, cancellation, or worker failures.
    pub fn push(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<Option<WakeCandidate>, AudioError> {
        if cancellation.is_cancelled() {
            self.stop_worker();
            return Err(cancelled());
        }
        if chunk.format() != STT_FORMAT {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "openWakeWord requires 16 kHz mono PCM",
            ));
        }
        self.start_worker(cancellation)?;
        let mut candidate = None;
        for sample in chunk.samples() {
            self.pending.push(*sample);
            if self.pending.len() == CHUNK_FRAMES {
                let score = self.infer(cancellation)?;
                self.pending.clear();
                if score >= self.config.threshold {
                    candidate = Some(WakeCandidate { score });
                }
            }
        }
        Ok(candidate)
    }

    /// Clears acoustic history after a candidate decision.
    ///
    /// # Errors
    ///
    /// Returns a redacted protocol or cancellation error.
    pub fn reset(&mut self, cancellation: &CancellationToken) -> Result<(), AudioError> {
        self.pending.clear();
        if self.worker.is_none() {
            return Ok(());
        }
        self.write_request(REQUEST_RESET, &[])?;
        match self.wait_frame(cancellation)? {
            WorkerFrame::Reset => Ok(()),
            WorkerFrame::Ready | WorkerFrame::Score(_) | WorkerFrame::Error => {
                self.stop_worker();
                Err(backend_error())
            }
        }
    }

    pub fn shutdown(&mut self) {
        self.stop_worker();
    }

    fn start_worker(&mut self, cancellation: &CancellationToken) -> Result<(), AudioError> {
        if self.worker.as_mut().is_some_and(Worker::has_exited) {
            self.stop_worker();
        }
        if self.worker.is_some() {
            return Ok(());
        }
        let mut command = Command::new(&self.config.python);
        command
            .arg("-u")
            .arg(&self.config.worker_script)
            .arg("--model")
            .arg(&self.config.model)
            .arg("--melspec")
            .arg(&self.config.melspectrogram)
            .arg("--embedding")
            .arg(&self.config.embedding)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
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
            if cancellation.is_cancelled() || Instant::now() >= deadline {
                self.stop_worker();
                return Err(if cancellation.is_cancelled() {
                    cancelled()
                } else {
                    backend_error()
                });
            }
            match self.worker_frame() {
                Ok(Some(WorkerFrame::Ready)) => return Ok(()),
                Ok(Some(WorkerFrame::Score(_) | WorkerFrame::Reset | WorkerFrame::Error))
                | Err(()) => {
                    self.stop_worker();
                    return Err(backend_error());
                }
                Ok(None) => {}
            }
        }
    }

    fn infer(&mut self, cancellation: &CancellationToken) -> Result<f32, AudioError> {
        let mut payload = Vec::with_capacity(CHUNK_BYTES);
        for sample in &self.pending {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        self.write_request(REQUEST_AUDIO, &payload)?;
        match self.wait_frame(cancellation)? {
            WorkerFrame::Score(score) if (0.0..=1.0).contains(&score) => Ok(score),
            WorkerFrame::Ready
            | WorkerFrame::Score(_)
            | WorkerFrame::Reset
            | WorkerFrame::Error => {
                self.stop_worker();
                Err(backend_error())
            }
        }
    }

    fn write_request(&mut self, kind: u8, payload: &[u8]) -> Result<(), AudioError> {
        let worker = self.worker.as_mut().ok_or_else(backend_error)?;
        let size = u32::try_from(payload.len()).map_err(|_| backend_error())?;
        if worker.stdin.write_all(&[kind]).is_err()
            || worker.stdin.write_all(&size.to_be_bytes()).is_err()
            || worker.stdin.write_all(payload).is_err()
            || worker.stdin.flush().is_err()
        {
            self.stop_worker();
            return Err(backend_error());
        }
        Ok(())
    }

    fn wait_frame(&mut self, cancellation: &CancellationToken) -> Result<WorkerFrame, AudioError> {
        loop {
            if cancellation.is_cancelled() {
                self.stop_worker();
                return Err(cancelled());
            }
            match self.worker_frame() {
                Ok(Some(frame)) => return Ok(frame),
                Ok(None) => {}
                Err(()) => {
                    self.stop_worker();
                    return Err(backend_error());
                }
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
        self.pending.clear();
    }
}

impl Drop for OpenWakeWordDetector {
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
    Ready,
    Score(f32),
    Reset,
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
        if size > 8 {
            let _ = sender.send(WorkerFrame::Error);
            return;
        }
        let mut payload = vec![0_u8; size];
        if reader.read_exact(&mut payload).is_err() {
            return;
        }
        let frame = match header[0] {
            FRAME_READY
                if payload
                    == [
                        0, 0, 62, 128, // 16_000 Hz
                        0, 0, 5, 0, // 1_280 frames
                    ] =>
            {
                WorkerFrame::Ready
            }
            FRAME_SCORE if payload.len() == 4 => WorkerFrame::Score(f32::from_be_bytes([
                payload[0], payload[1], payload[2], payload[3],
            ])),
            FRAME_RESET if payload.is_empty() => WorkerFrame::Reset,
            _ => WorkerFrame::Error,
        };
        if sender.send(frame).is_err() {
            return;
        }
    }
}

fn backend_error() -> AudioError {
    AudioError::new(AudioErrorKind::Backend, "openWakeWord worker failed")
}

fn cancelled() -> AudioError {
    AudioError::new(AudioErrorKind::Cancelled, "wake detection was cancelled")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::mpsc::sync_channel;
    use std::time::Duration;

    use oreo_core::CancellationToken;

    use crate::{PcmChunk, STT_FORMAT};

    use super::{
        FRAME_READY, FRAME_SCORE, OpenWakeWordConfig, OpenWakeWordDetector, WorkerFrame,
        read_frames,
    };

    #[test]
    fn framed_scores_are_bounded_before_decoding() {
        let mut wire = vec![FRAME_READY, 0, 0, 0, 8, 0, 0, 62, 128, 0, 0, 5, 0];
        wire.extend_from_slice(&[FRAME_SCORE, 0, 0, 0, 4]);
        wire.extend_from_slice(&0.25_f32.to_be_bytes());
        let (sender, receiver) = sync_channel(2);
        read_frames(Cursor::new(wire), &sender);
        assert!(matches!(receiver.recv(), Ok(WorkerFrame::Ready)));
        assert!(matches!(receiver.recv(), Ok(WorkerFrame::Score(0.25))));
    }

    #[test]
    fn fake_worker_buffers_exact_model_windows_and_resets() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let script = temporary.path().join("fake-worker.py");
        fs::write(
            &script,
            r"import struct, sys
source = sys.stdin.buffer
out = sys.stdout.buffer
def frame(kind, payload=b''):
    out.write(bytes([kind]) + struct.pack('>I', len(payload)) + payload)
    out.flush()
frame(1, struct.pack('>II', 16000, 1280))
while True:
    header = source.read(5)
    if not header:
        break
    kind, size = header[0], struct.unpack('>I', header[1:])[0]
    payload = source.read(size)
    if len(payload) != size:
        break
    if kind == 1:
        frame(2, struct.pack('>f', 0.75))
    elif kind == 2:
        frame(3)
",
        )
        .expect("fake worker script");
        let model = temporary.path().join("model.onnx");
        let melspec = temporary.path().join("melspectrogram.onnx");
        let embedding = temporary.path().join("embedding.onnx");
        for path in [&model, &melspec, &embedding] {
            fs::write(path, b"fixture").expect("fixture model");
        }
        let config = OpenWakeWordConfig {
            python: std::env::var_os("PYTHON")
                .map_or_else(|| PathBuf::from("/usr/bin/python3"), PathBuf::from),
            worker_script: script,
            model,
            melspectrogram: melspec,
            embedding,
            threshold: 0.5,
            startup_timeout: Duration::from_secs(2),
        };
        let mut detector = OpenWakeWordDetector::new(config).expect("detector config is valid");
        let cancellation = CancellationToken::new();
        let chunk = PcmChunk::new(STT_FORMAT, vec![0; 320]).expect("chunk is valid");
        for _ in 0..3 {
            assert_eq!(
                detector.push(&chunk, &cancellation).expect("worker runs"),
                None
            );
        }
        assert_eq!(
            detector.push(&chunk, &cancellation).expect("worker runs"),
            Some(super::WakeCandidate { score: 0.75 })
        );
        detector.reset(&cancellation).expect("worker resets");
        detector.shutdown();
    }
}
