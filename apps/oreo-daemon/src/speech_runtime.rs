use std::collections::{HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use oreo_audio::{
    AudioErrorKind, AudioLimits, AudioOutput, CpalOutput, PcmConverter, PocketTtsConfig,
    PocketTtsSynthesizer, StreamingSynthesizer, normalize_for_speech,
};
use oreo_core::CancellationToken;

const SPEECH_QUEUE_CAPACITY: usize = 16;
const ECHO_REFERENCE_CAPACITY: usize = 8;
const ECHO_TAIL: Duration = Duration::from_millis(750);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpeechRuntimeEvent {
    Ready,
    Generating,
    Started,
    Finished,
    Cancelled,
    Failed,
}

struct SpeechRequest {
    text: String,
    cancellation: CancellationToken,
}

#[derive(Clone)]
pub(crate) struct SpeechIngress {
    sender: SyncSender<SpeechRequest>,
    interrupt: Arc<AtomicBool>,
    speaking: Arc<AtomicBool>,
    echo: Arc<Mutex<PlaybackEchoGuard>>,
}

impl SpeechIngress {
    pub(crate) fn speak(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), SpeechRuntimeError> {
        let text = normalize_for_speech(text, AudioLimits::sbc())
            .map_err(|_| SpeechRuntimeError::new("speech text is invalid"))?;
        let reference = text.clone();
        match self.sender.try_send(SpeechRequest {
            text,
            cancellation: cancellation.clone(),
        }) {
            Ok(()) => {
                if let Ok(mut echo) = self.echo.lock() {
                    echo.remember(&reference);
                }
                Ok(())
            }
            Err(TrySendError::Full(_)) => Err(SpeechRuntimeError::new("speech queue is full")),
            Err(TrySendError::Disconnected(_)) => {
                Err(SpeechRuntimeError::new("speech runtime is unavailable"))
            }
        }
    }

    pub(crate) fn cancel(&self, cancellation: &CancellationToken) {
        cancellation.cancel();
        self.interrupt.store(true, Ordering::Release);
    }

    pub(crate) fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::Acquire)
    }

    pub(crate) fn resembles_output(&self, transcript: &str) -> bool {
        self.echo
            .lock()
            .is_ok_and(|echo| echo.resembles(transcript, self.is_speaking()))
    }
}

pub(crate) struct SpeechRuntime {
    ingress: SpeechIngress,
    stopping: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl SpeechRuntime {
    pub(crate) fn spawn(
        repository_root: &Path,
        emit: fn(SpeechRuntimeEvent),
    ) -> Result<Self, SpeechRuntimeError> {
        let limits = AudioLimits::sbc();
        let synthesizer =
            PocketTtsSynthesizer::new(PocketTtsConfig::for_repository(repository_root), limits)
                .map_err(|_| {
                    SpeechRuntimeError::new("speech synthesizer configuration is invalid")
                })?;
        let output = CpalOutput::open_default(limits)
            .map_err(|_| SpeechRuntimeError::new("speaker could not be opened"))?;
        let (sender, receiver) = mpsc::sync_channel(SPEECH_QUEUE_CAPACITY);
        let interrupt = Arc::new(AtomicBool::new(false));
        let speaking = Arc::new(AtomicBool::new(false));
        let echo = Arc::new(Mutex::new(PlaybackEchoGuard::default()));
        let ingress = SpeechIngress {
            sender,
            interrupt: interrupt.clone(),
            speaking: speaking.clone(),
            echo: echo.clone(),
        };
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = stopping.clone();
        let loop_context = SpeechLoopContext {
            receiver,
            stopping: worker_stopping,
            interrupt,
            speaking,
            echo,
            emit,
        };
        let handle = thread::Builder::new()
            .name("oreo-speech".to_owned())
            .spawn(move || {
                speech_loop(synthesizer, output, &loop_context);
            })
            .map_err(|_| SpeechRuntimeError::new("speech thread could not start"))?;
        Ok(Self {
            ingress,
            stopping,
            handle: Some(handle),
        })
    }

    pub(crate) fn ingress(&self) -> SpeechIngress {
        self.ingress.clone()
    }

    pub(crate) fn shutdown(mut self) -> Result<(), SpeechRuntimeError> {
        self.stopping.store(true, Ordering::Release);
        drop(self.ingress);
        self.handle
            .take()
            .ok_or_else(|| SpeechRuntimeError::new("speech thread is unavailable"))?
            .join()
            .map_err(|_| SpeechRuntimeError::new("speech thread stopped unexpectedly"))
    }
}

struct SpeechLoopContext {
    receiver: Receiver<SpeechRequest>,
    stopping: Arc<AtomicBool>,
    interrupt: Arc<AtomicBool>,
    speaking: Arc<AtomicBool>,
    echo: Arc<Mutex<PlaybackEchoGuard>>,
    emit: fn(SpeechRuntimeEvent),
}

fn speech_loop(
    mut synthesizer: PocketTtsSynthesizer,
    mut output: CpalOutput,
    context: &SpeechLoopContext,
) {
    let startup = CancellationToken::new();
    if synthesizer.prewarm(&startup).is_err() {
        (context.emit)(SpeechRuntimeEvent::Failed);
        return;
    }
    let native_format = output.native_format();
    if output.begin(native_format).is_err() {
        (context.emit)(SpeechRuntimeEvent::Failed);
        return;
    }
    let mut output_active = true;
    (context.emit)(SpeechRuntimeEvent::Ready);
    while !context.stopping.load(Ordering::Acquire) {
        if context.interrupt.swap(false, Ordering::AcqRel) {
            output.stop();
            output_active = false;
            context.speaking.store(false, Ordering::Release);
            (context.emit)(SpeechRuntimeEvent::Cancelled);
        }
        let request = match context.receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(request) => request,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if context.speaking.load(Ordering::Acquire) && output.stats().queued_samples == 0 {
                    context.speaking.store(false, Ordering::Release);
                    if let Ok(mut echo) = context.echo.lock() {
                        echo.mark_finished();
                    }
                    (context.emit)(SpeechRuntimeEvent::Finished);
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if request.cancellation.is_cancelled() {
            continue;
        }
        let was_idle = !context.speaking.swap(true, Ordering::AcqRel);
        (context.emit)(SpeechRuntimeEvent::Generating);
        if !output_active && output.begin(native_format).is_err() {
            (context.emit)(SpeechRuntimeEvent::Failed);
            continue;
        }
        output_active = true;
        let mut converter = None;
        let mut playback_started = false;
        let mut lead_in_pending = was_idle;
        let result = synthesizer.synthesize(&request.text, &request.cancellation, &mut |chunk| {
            let converter = match &mut converter {
                Some(converter) => converter,
                None => converter.insert(PcmConverter::new(
                    chunk.format(),
                    native_format,
                    AudioLimits::sbc(),
                )?),
            };
            converter.push(&chunk, &request.cancellation, &mut |converted| {
                write_playback_chunk(
                    &mut output,
                    &converted,
                    &request.cancellation,
                    &mut lead_in_pending,
                    &mut playback_started,
                    context.emit,
                )
            })
        });
        let result = result.and_then(|()| {
            if let Some(converter) = &mut converter {
                converter.finish(&request.cancellation, &mut |converted| {
                    write_playback_chunk(
                        &mut output,
                        &converted,
                        &request.cancellation,
                        &mut lead_in_pending,
                        &mut playback_started,
                        context.emit,
                    )
                })?;
            }
            Ok(())
        });
        match result {
            Ok(()) => {}
            Err(error) if error.kind == AudioErrorKind::Cancelled => {
                output.stop();
                output_active = false;
                context.speaking.store(false, Ordering::Release);
                (context.emit)(SpeechRuntimeEvent::Cancelled);
            }
            Err(_) => {
                output.stop();
                output_active = false;
                context.speaking.store(false, Ordering::Release);
                (context.emit)(SpeechRuntimeEvent::Failed);
            }
        }
    }
    output.stop();
    context.speaking.store(false, Ordering::Release);
    synthesizer.shutdown();
}

#[derive(Default)]
struct PlaybackEchoGuard {
    references: VecDeque<HashSet<String>>,
    tail_until: Option<Instant>,
}

impl PlaybackEchoGuard {
    fn remember(&mut self, text: &str) {
        let words = lexical_words(text);
        if words.is_empty() {
            return;
        }
        if self.references.len() == ECHO_REFERENCE_CAPACITY {
            self.references.pop_front();
        }
        self.references.push_back(words);
        self.tail_until = None;
    }

    fn mark_finished(&mut self) {
        self.tail_until = Some(Instant::now() + ECHO_TAIL);
    }

    fn resembles(&self, transcript: &str, speaking: bool) -> bool {
        if !speaking
            && self
                .tail_until
                .is_none_or(|deadline| Instant::now() > deadline)
        {
            return false;
        }
        let heard = lexical_words(transcript);
        if heard.is_empty() {
            return false;
        }
        self.references.iter().any(|reference| {
            let shared = heard.intersection(reference).count();
            let smaller = heard.len().min(reference.len());
            shared == smaller && smaller == 1 || smaller >= 2 && shared * 4 >= smaller * 3
        })
    }
}

fn lexical_words(text: &str) -> HashSet<String> {
    text.chars()
        .flat_map(char::to_lowercase)
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn write_with_backpressure(
    output: &mut CpalOutput,
    chunk: &oreo_audio::PcmChunk,
    cancellation: &CancellationToken,
) -> Result<(), oreo_audio::AudioError> {
    loop {
        match output.write(chunk, cancellation) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind == AudioErrorKind::Capacity => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(error),
        }
    }
}

fn write_playback_chunk(
    output: &mut CpalOutput,
    chunk: &oreo_audio::PcmChunk,
    cancellation: &CancellationToken,
    lead_in_pending: &mut bool,
    playback_started: &mut bool,
    emit: fn(SpeechRuntimeEvent),
) -> Result<(), oreo_audio::AudioError> {
    let played_before = output.stats().content_samples_played;
    if *lead_in_pending {
        let format = chunk.format();
        let frames = usize::try_from(format.sample_rate_hz)
            .unwrap_or(usize::MAX)
            .saturating_mul(60)
            / 1_000;
        let samples = frames.saturating_mul(usize::from(format.channels));
        let silence = oreo_audio::PcmChunk::new(format, vec![0; samples])?;
        write_with_backpressure(output, &silence, cancellation)?;
        *lead_in_pending = false;
    }
    write_with_backpressure(output, chunk, cancellation)?;
    if !*playback_started {
        wait_for_playback(output, played_before, cancellation)?;
        *playback_started = true;
        emit(SpeechRuntimeEvent::Started);
    }
    Ok(())
}

fn wait_for_playback(
    output: &CpalOutput,
    played_before: u64,
    cancellation: &CancellationToken,
) -> Result<(), oreo_audio::AudioError> {
    let deadline = Instant::now() + Duration::from_millis(500);
    while output.stats().content_samples_played == played_before {
        if cancellation.is_cancelled() {
            return Err(oreo_audio::AudioError::new(
                AudioErrorKind::Cancelled,
                "audio playback was cancelled",
            ));
        }
        if Instant::now() >= deadline {
            return Err(oreo_audio::AudioError::new(
                AudioErrorKind::Backend,
                "speaker did not consume audio",
            ));
        }
        thread::sleep(Duration::from_millis(5));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SpeechRuntimeError {
    message: &'static str,
}

impl SpeechRuntimeError {
    const fn new(message: &'static str) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for SpeechRuntimeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for SpeechRuntimeError {}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::PlaybackEchoGuard;

    #[test]
    fn echo_guard_rejects_output_but_preserves_barge_in() {
        let mut guard = PlaybackEchoGuard::default();
        guard.remember("The front door is locked and the kitchen light is off.");

        assert!(guard.resembles("the front door is locked", true));
        assert!(!guard.resembles("Oreo stop and set a timer", true));
    }

    #[test]
    fn echo_guard_has_only_a_short_post_playback_tail() {
        let mut guard = PlaybackEchoGuard::default();
        guard.remember("Your timer is now running.");
        guard.mark_finished();
        assert!(guard.resembles("your timer is running", false));
        guard.tail_until = std::time::Instant::now().checked_sub(Duration::from_millis(1));
        assert!(!guard.resembles("your timer is running", false));
    }
}
