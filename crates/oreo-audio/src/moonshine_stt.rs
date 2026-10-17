use std::fmt;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use oreo_core::CancellationToken;

use crate::{
    AmbientTranscriber, AudioError, AudioErrorKind, AudioFormat, AudioLimits, PcmChunk, STT_FORMAT,
    StreamingTranscriber, TranscriptionConfidence,
};

const COMMAND_BEGIN: u8 = 1;
const COMMAND_AUDIO: u8 = 2;
const COMMAND_FINISH: u8 = 3;
const COMMAND_ABORT: u8 = 4;
const FRAME_READY: u8 = 1;
const FRAME_ACK: u8 = 2;
const FRAME_PARTIAL: u8 = 3;
const FRAME_FINAL: u8 = 4;
const FRAME_ERROR: u8 = 5;
const MAX_AUDIO_BYTES: usize = 32_000;
const POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Clone, Debug)]
pub struct MoonshineSttConfig {
    pub python: PathBuf,
    pub worker_script: PathBuf,
    pub model_directory: PathBuf,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
}

impl MoonshineSttConfig {
    #[must_use]
    pub fn for_repository(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self {
            python: root.join(".venv-stt/bin/python"),
            worker_script: root.join("scripts/moonshine-stt-worker.py"),
            model_directory: root.join(
                "models/cache/moonshine-voice/download.moonshine.ai/model/\
                 small-streaming-en/quantized_26_08_21",
            ),
            startup_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(5),
        }
    }

    fn validate(&self) -> Result<(), AudioError> {
        if !self.python.is_file()
            || !self.worker_script.is_file()
            || !self.model_directory.is_dir()
            || !(Duration::from_secs(1)..=Duration::from_mins(1)).contains(&self.startup_timeout)
            || !(Duration::from_millis(100)..=Duration::from_secs(30))
                .contains(&self.request_timeout)
        {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "Moonshine worker configuration is invalid",
            ));
        }
        Ok(())
    }
}

pub struct MoonshineTranscriber {
    config: MoonshineSttConfig,
    worker: Option<Worker>,
    active: bool,
    attention_name: Option<String>,
    attention_detected: bool,
    live_transcript: String,
    max_transcript_bytes: usize,
}

impl fmt::Debug for MoonshineTranscriber {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MoonshineTranscriber")
            .field("worker_active", &self.worker.is_some())
            .field("utterance_active", &self.active)
            .field("transcript_bytes", &self.live_transcript.len())
            .field("max_transcript_bytes", &self.max_transcript_bytes)
            .finish_non_exhaustive()
    }
}

impl MoonshineTranscriber {
    /// Creates a lazy persistent Moonshine adapter.
    ///
    /// # Errors
    ///
    /// Rejects missing worker assets, model files, or invalid audio limits.
    pub fn load(config: MoonshineSttConfig, limits: AudioLimits) -> Result<Self, AudioError> {
        config.validate()?;
        let limits = limits.validate()?;
        Ok(Self {
            config,
            worker: None,
            active: false,
            attention_name: None,
            attention_detected: false,
            live_transcript: String::with_capacity(limits.max_transcript_bytes),
            max_transcript_bytes: limits.max_transcript_bytes,
        })
    }

    fn start_worker(&mut self) -> Result<(), AudioError> {
        if self.worker.as_mut().is_some_and(Worker::has_exited) {
            self.stop_worker();
        }
        if self.worker.is_some() {
            return Ok(());
        }
        let attention_name = self.attention_name.as_deref().ok_or_else(|| {
            AudioError::new(
                AudioErrorKind::InvalidConfig,
                "Moonshine attention name is not configured",
            )
        })?;
        let mut child = Command::new(&self.config.python)
            .arg("-u")
            .arg(&self.config.worker_script)
            .arg("--model")
            .arg(&self.config.model_directory)
            .arg("--attention-name")
            .arg(attention_name)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("OMP_NUM_THREADS", "2")
            .env("MKL_NUM_THREADS", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| backend_error())?;
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
        if self.wait_for(self.config.startup_timeout, |frame| {
            matches!(frame, WorkerFrame::Ready)
        })? {
            Ok(())
        } else {
            self.stop_worker();
            Err(backend_error())
        }
    }

    fn send(&mut self, command: u8, payload: &[u8]) -> Result<(), AudioError> {
        if payload.len() > MAX_AUDIO_BYTES {
            return Err(AudioError::new(
                AudioErrorKind::Capacity,
                "Moonshine audio frame is too large",
            ));
        }
        let size = u32::try_from(payload.len()).map_err(|_| backend_error())?;
        let worker = self.worker.as_mut().ok_or_else(backend_error)?;
        worker
            .stdin
            .write_all(&[command])
            .and_then(|()| worker.stdin.write_all(&size.to_be_bytes()))
            .and_then(|()| worker.stdin.write_all(payload))
            .and_then(|()| worker.stdin.flush())
            .map_err(|_| backend_error())
    }

    fn wait_for(
        &mut self,
        timeout: Duration,
        accepted: impl Fn(&WorkerFrame) -> bool,
    ) -> Result<bool, AudioError> {
        let deadline = Instant::now() + timeout;
        loop {
            if Instant::now() >= deadline {
                return Ok(false);
            }
            let frame = match self
                .worker
                .as_ref()
                .ok_or_else(backend_error)?
                .receiver
                .recv_timeout(POLL_INTERVAL)
            {
                Ok(frame) => frame,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return Err(backend_error()),
            };
            if accepted(&frame) {
                return Ok(true);
            }
            self.apply_frame(frame)?;
        }
    }

    fn apply_frame(&mut self, frame: WorkerFrame) -> Result<(), AudioError> {
        match frame {
            WorkerFrame::Partial(text) | WorkerFrame::Final(text) => {
                if text.len() > self.max_transcript_bytes {
                    return Err(AudioError::new(
                        AudioErrorKind::Capacity,
                        "Moonshine transcript is too large",
                    ));
                }
                self.attention_detected |= self
                    .attention_name
                    .as_deref()
                    .is_some_and(|name| contains_attention_name(&text, name));
                self.live_transcript = text;
                Ok(())
            }
            WorkerFrame::Error | WorkerFrame::Ready | WorkerFrame::Ack => Err(backend_error()),
        }
    }

    fn drain_frames(&mut self) -> Result<(), AudioError> {
        loop {
            let frame = match self
                .worker
                .as_ref()
                .ok_or_else(backend_error)?
                .receiver
                .try_recv()
            {
                Ok(frame) => frame,
                Err(TryRecvError::Empty) => return Ok(()),
                Err(TryRecvError::Disconnected) => return Err(backend_error()),
            };
            self.apply_frame(frame)?;
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
        self.active = false;
    }

    fn reset_utterance(&mut self) {
        self.active = false;
        self.attention_detected = false;
        self.live_transcript.clear();
    }
}

fn contains_attention_name(text: &str, name: &str) -> bool {
    text.split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .any(|word| {
            word.eq_ignore_ascii_case(name)
                || (name.len() >= 5
                    && word.len().abs_diff(name.len()) <= 1
                    && consonant_skeleton(word).eq(consonant_skeleton(name))
                    && edit_distance_with_limit(&word.to_ascii_lowercase(), name, 3).is_some())
        })
}

fn consonant_skeleton(word: &str) -> impl Iterator<Item = u8> + '_ {
    word.bytes()
        .map(|byte| byte.to_ascii_lowercase())
        .filter(|byte| byte.is_ascii_alphabetic() && !b"aeiouy".contains(byte))
}

fn edit_distance_with_limit(left: &str, right: &str, limit: usize) -> Option<usize> {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len().abs_diff(right.len()) > limit {
        return None;
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_byte) in left.iter().enumerate() {
        current[0] = left_index + 1;
        let mut row_minimum = current[0];
        for (right_index, right_byte) in right.iter().enumerate() {
            current[right_index + 1] = (previous[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(previous[right_index] + usize::from(left_byte != right_byte));
            row_minimum = row_minimum.min(current[right_index + 1]);
        }
        if row_minimum > limit {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[right.len()] <= limit).then_some(previous[right.len()])
}

impl StreamingTranscriber for MoonshineTranscriber {
    fn begin(&mut self, format: AudioFormat) -> Result<(), AudioError> {
        if self.active {
            return Err(AudioError::new(
                AudioErrorKind::InvalidTransition,
                "Moonshine transcription is already active",
            ));
        }
        if format != STT_FORMAT {
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "Moonshine requires 16 kHz mono PCM",
            ));
        }
        self.start_worker()?;
        self.reset_utterance();
        self.send(COMMAND_BEGIN, &[])?;
        if !self.wait_for(self.config.request_timeout, |frame| {
            matches!(frame, WorkerFrame::Ack)
        })? {
            self.stop_worker();
            return Err(backend_error());
        }
        self.active = true;
        Ok(())
    }

    fn push(
        &mut self,
        chunk: &PcmChunk,
        cancellation: &CancellationToken,
    ) -> Result<(), AudioError> {
        if cancellation.is_cancelled() {
            self.abort();
            return Err(cancelled());
        }
        if !self.active {
            return Err(AudioError::new(
                AudioErrorKind::InvalidTransition,
                "Moonshine transcription is not active",
            ));
        }
        if chunk.format() != STT_FORMAT {
            self.abort();
            return Err(AudioError::new(
                AudioErrorKind::UnsupportedFormat,
                "Moonshine requires 16 kHz mono PCM",
            ));
        }
        self.drain_frames()?;
        let mut payload = Vec::with_capacity(chunk.samples().len() * 2);
        for sample in chunk.samples() {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        if let Err(error) = self.send(COMMAND_AUDIO, &payload) {
            self.stop_worker();
            return Err(error);
        }
        self.drain_frames()
    }

    fn abort(&mut self) {
        if self.active
            && (self.send(COMMAND_ABORT, &[]).is_err()
                || !matches!(
                    self.wait_for(self.config.request_timeout, |frame| {
                        matches!(frame, WorkerFrame::Ack)
                    }),
                    Ok(true)
                ))
        {
            self.stop_worker();
        }
        self.reset_utterance();
    }

    fn finish(&mut self, cancellation: &CancellationToken) -> Result<String, AudioError> {
        if cancellation.is_cancelled() {
            self.abort();
            return Err(cancelled());
        }
        if !self.active {
            return Err(AudioError::new(
                AudioErrorKind::InvalidTransition,
                "Moonshine transcription is not active",
            ));
        }
        self.send(COMMAND_FINISH, &[])?;
        let deadline = Instant::now() + self.config.request_timeout;
        loop {
            if cancellation.is_cancelled() {
                self.stop_worker();
                self.reset_utterance();
                return Err(cancelled());
            }
            if Instant::now() >= deadline {
                self.stop_worker();
                self.reset_utterance();
                return Err(backend_error());
            }
            let frame = match self
                .worker
                .as_ref()
                .ok_or_else(backend_error)?
                .receiver
                .recv_timeout(POLL_INTERVAL)
            {
                Ok(frame) => frame,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    self.stop_worker();
                    return Err(backend_error());
                }
            };
            match frame {
                WorkerFrame::Final(text) => {
                    self.apply_frame(WorkerFrame::Final(text))?;
                    self.active = false;
                    return Ok(self.live_transcript.clone());
                }
                other => self.apply_frame(other)?,
            }
        }
    }
}

impl AmbientTranscriber for MoonshineTranscriber {
    fn configure_attention_spotter(&mut self, name: &str) -> Result<(), AudioError> {
        if self.active || self.worker.is_some() {
            return Err(AudioError::new(
                AudioErrorKind::InvalidTransition,
                "Moonshine attention name cannot change after startup",
            ));
        }
        let normalized = name.trim().to_ascii_lowercase();
        if normalized.len() < 3
            || normalized.len() > 24
            || !normalized
                .chars()
                .all(|character| character.is_ascii_alphabetic())
        {
            return Err(AudioError::new(
                AudioErrorKind::InvalidConfig,
                "attention name must be one short word",
            ));
        }
        self.attention_name = Some(normalized);
        Ok(())
    }

    fn live_transcript(&mut self) -> Result<String, AudioError> {
        if !self.active {
            return Err(AudioError::new(
                AudioErrorKind::InvalidTransition,
                "Moonshine transcription is not active",
            ));
        }
        self.drain_frames()?;
        Ok(self.live_transcript.clone())
    }

    fn confidence(&self) -> Option<TranscriptionConfidence> {
        None
    }

    fn attention_detected(&self) -> bool {
        !self.active && self.attention_detected
    }

    fn suspend(&mut self) {
        self.stop_worker();
        self.reset_utterance();
    }
}

impl Drop for MoonshineTranscriber {
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
    Ack,
    Partial(String),
    Final(String),
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
        if size > 4_096 {
            let _ = sender.send(WorkerFrame::Error);
            return;
        }
        let mut payload = vec![0_u8; size];
        if reader.read_exact(&mut payload).is_err() {
            return;
        }
        let frame = match header[0] {
            FRAME_READY if payload.is_empty() => WorkerFrame::Ready,
            FRAME_ACK if payload.is_empty() => WorkerFrame::Ack,
            FRAME_PARTIAL => {
                String::from_utf8(payload).map_or(WorkerFrame::Error, WorkerFrame::Partial)
            }
            FRAME_FINAL => {
                String::from_utf8(payload).map_or(WorkerFrame::Error, WorkerFrame::Final)
            }
            FRAME_ERROR if payload.is_empty() => WorkerFrame::Error,
            _ => WorkerFrame::Error,
        };
        if sender.send(frame).is_err() {
            return;
        }
    }
}

fn backend_error() -> AudioError {
    AudioError::new(AudioErrorKind::Backend, "Moonshine STT worker failed")
}

fn cancelled() -> AudioError {
    AudioError::new(
        AudioErrorKind::Cancelled,
        "Moonshine transcription was cancelled",
    )
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::Duration;

    use oreo_core::CancellationToken;

    use crate::{AmbientTranscriber, AudioLimits, PcmChunk, STT_FORMAT, StreamingTranscriber};

    use super::{MoonshineSttConfig, MoonshineTranscriber, contains_attention_name};

    #[test]
    fn attention_matching_accepts_close_model_rendering_but_not_common_collision() {
        assert!(contains_attention_name("Hey, Orion", "orion"));
        assert!(contains_attention_name("Hey Aureon explain this", "orion"));
        assert!(!contains_attention_name("play the audio", "orion"));
        assert!(!contains_attention_name("that was an ordeal", "orion"));
    }

    #[test]
    fn fake_worker_is_reused_across_streaming_utterances() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let model = temporary.path().join("model");
        fs::create_dir(&model).expect("model directory");
        let script = temporary.path().join("fake-moonshine.py");
        fs::write(
            &script,
            r"import struct, sys
source = sys.stdin.buffer
output = sys.stdout.buffer
def frame(kind, payload=b''):
    output.write(bytes([kind]) + struct.pack('>I', len(payload)) + payload)
    output.flush()
def command():
    header = source.read(5)
    if not header:
        return None, b''
    size = struct.unpack('>I', header[1:])[0]
    return header[0], source.read(size)
frame(1)
while True:
    kind, payload = command()
    if kind is None:
        break
    if kind == 1:
        frame(2)
    elif kind == 2:
        frame(3, b'hello orion')
    elif kind == 3:
        frame(4, b'hello orion')
    elif kind == 4:
        frame(2)
",
        )
        .expect("fake worker script");
        let python = std::env::var_os("PYTHON")
            .map_or_else(|| PathBuf::from("/usr/bin/python3"), PathBuf::from);
        let config = MoonshineSttConfig {
            python,
            worker_script: script,
            model_directory: model,
            startup_timeout: Duration::from_secs(2),
            request_timeout: Duration::from_secs(2),
        };
        let mut transcriber =
            MoonshineTranscriber::load(config, AudioLimits::sbc()).expect("adapter config");
        transcriber
            .configure_attention_spotter("orion")
            .expect("attention config");
        let cancellation = CancellationToken::new();
        for _ in 0..2 {
            transcriber.begin(STT_FORMAT).expect("begin utterance");
            transcriber
                .push(
                    &PcmChunk::new(STT_FORMAT, vec![1; 320]).expect("PCM chunk"),
                    &cancellation,
                )
                .expect("push audio");
            assert_eq!(
                transcriber.finish(&cancellation).expect("final text"),
                "hello orion"
            );
            assert!(transcriber.attention_detected());
        }
    }
}
