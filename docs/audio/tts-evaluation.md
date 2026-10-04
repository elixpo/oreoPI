# PocketTTS laptop evaluation

Status: PocketTTS reference passed; sherpa PocketTTS rejected; Kitten evaluation pending

The pinned PocketTTS 3.3.0 Python reference passed the laptop latency,
throughput, offline-cache, and cancellation checks with dynamic int8 enabled.
The offline run is the representative warm-cache result:

| Metric | Result | Gate |
|---|---:|---:|
| Warm first-audio p95 | 105 ms | below 500 ms |
| Median generation speed | 8.50x real time | above 1.0x |
| Cached model load | 1,021 ms | reported, not a turn-path gate |
| Cached voice load | 2 ms | reported, not a turn-path gate |
| Cancellation after first chunk | 60 ms | must stop cleanly |
| Loaded RSS | 960,784 KiB | whole-device hard limit is 1,200 MiB |
| Peak RSS | 1,055,640 KiB | whole-device hard limit is 1,200 MiB |

The initial authenticated run had a 211 ms first-audio p95 and 8.37x median
generation speed. Its long model-load measurement includes the first gated
artefact download and is not a cold-start result. The repeat with network access
disabled proves all inference artefacts were present locally.

The Python reference is not selected for the device runtime despite its strong
speed. Roughly one GiB for TTS alone leaves too little room for Oreo, Vosk, and
the operating system under the current 1,200 MiB hard ceiling. It remains the
quality and performance reference for laptop development.

The next candidate is sherpa-onnx 1.13.8 with its published 98 MB PocketTTS
int8 archive. It exposes PocketTTS through the official Rust API, supports
incremental cancellable callbacks, and has an AArch64 deployment path. Its
converted January 2026 model is not identical to PocketTTS 3.3.0's September
English weights, so generated voice quality must be listened to rather than
inferred from the Python result. The bundled reference voice is benchmark-only
until its provenance is reviewed; it is not an Oreo product voice selection.

The initial sherpa comparison accidentally used five flow steps from its Python
API example. It loaded at 351,912 KiB RSS and peaked at 628,736 KiB, but its
1,539 ms first-audio p95 failed the 500 ms gate. Current upstream PocketTTS
instructions use two steps and identify this setting as the quality/speed
tradeoff, so the selection measurement is repeated at two. The five-step
result remains recorded as evidence rather than being presented as the runtime
default.

The official two-step configuration still measured 1,624 ms first-audio p95
despite a much better 353,452 KiB loaded RSS and 629,572 KiB peak. The operator
also found the start of every utterance blurry and the overall voice worse than
the Python reference. Sherpa PocketTTS is therefore rejected as Oreo's primary;
one flow step is not evaluated because it explicitly trades away more quality
and cannot repair the first-word defect.

The next low-resource candidate is Kitten TTS Nano 0.8 int8. It is English-only,
has 15 million parameters and eight voices, uses the same sherpa-onnx Rust API,
and has a 31 MB published archive. This is a better fit for Oreo's least-weight
requirement than immediately testing the 129 MB, 31-language Supertonic model.
PocketTTS remains the laptop quality reference and proof that WP-004's
PocketTTS streaming path works; the device backend remains replaceable.
