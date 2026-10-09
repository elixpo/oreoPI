use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread;
use std::time::Duration;

use oreo_audio::{
    AudioErrorKind, AudioLimits, AudioOutput, CpalOutput, PcmConverter, PocketTtsConfig,
    PocketTtsSynthesizer, StreamingSynthesizer, normalize_for_speech,
};
use oreo_core::CancellationToken;

const SPEECH_QUEUE_CAPACITY: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpeechRuntimeEvent {
    Ready,
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
}

impl SpeechIngress {
    pub(crate) fn speak(
        &self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<(), SpeechRuntimeError> {
        let text = normalize_for_speech(text, AudioLimits::sbc())
            .map_err(|_| SpeechRuntimeError::new("speech text is invalid"))?;
        match self.sender.try_send(SpeechRequest {
            text,
            cancellation: cancellation.clone(),
        }) {
            Ok(()) => Ok(()),
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
        let ingress = SpeechIngress {
            sender,
            interrupt: interrupt.clone(),
            speaking: speaking.clone(),
        };
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = stopping.clone();
        let handle = thread::Builder::new()
            .name("oreo-speech".to_owned())
            .spawn(move || {
                speech_loop(
                    synthesizer,
                    output,
                    &receiver,
                    &worker_stopping,
                    &interrupt,
                    &speaking,
                    emit,
                );
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

fn speech_loop(
    mut synthesizer: PocketTtsSynthesizer,
    mut output: CpalOutput,
    receiver: &Receiver<SpeechRequest>,
    stopping: &AtomicBool,
    interrupt: &AtomicBool,
    speaking: &AtomicBool,
    emit: fn(SpeechRuntimeEvent),
) {
    let startup = CancellationToken::new();
    if synthesizer.prewarm(&startup).is_err() {
        emit(SpeechRuntimeEvent::Failed);
        return;
    }
    let native_format = output.native_format();
    if output.begin(native_format).is_err() {
        emit(SpeechRuntimeEvent::Failed);
        return;
    }
    let mut output_active = true;
    emit(SpeechRuntimeEvent::Ready);
    while !stopping.load(Ordering::Acquire) {
        if interrupt.swap(false, Ordering::AcqRel) {
            output.stop();
            output_active = false;
            speaking.store(false, Ordering::Release);
            emit(SpeechRuntimeEvent::Cancelled);
        }
        let request = match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(request) => request,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if request.cancellation.is_cancelled() {
            continue;
        }
        speaking.store(true, Ordering::Release);
        emit(SpeechRuntimeEvent::Started);
        if !output_active && output.begin(native_format).is_err() {
            emit(SpeechRuntimeEvent::Failed);
            continue;
        }
        output_active = true;
        let mut converter = None;
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
                write_with_backpressure(&mut output, &converted, &request.cancellation)
            })
        });
        let result = result.and_then(|()| {
            if let Some(converter) = &mut converter {
                converter.finish(&request.cancellation, &mut |converted| {
                    write_with_backpressure(&mut output, &converted, &request.cancellation)
                })?;
            }
            Ok(())
        });
        match result {
            Ok(()) => emit(SpeechRuntimeEvent::Finished),
            Err(error) if error.kind == AudioErrorKind::Cancelled => {
                output.stop();
                output_active = false;
                speaking.store(false, Ordering::Release);
                emit(SpeechRuntimeEvent::Cancelled);
            }
            Err(_) => {
                output.stop();
                output_active = false;
                speaking.store(false, Ordering::Release);
                emit(SpeechRuntimeEvent::Failed);
            }
        }
    }
    output.stop();
    speaking.store(false, Ordering::Release);
    synthesizer.shutdown();
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
