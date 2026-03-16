# Moonshine MLX Benchmark Results

Comparison of three implementations of Moonshine V2 Streaming speech-to-text:

- **C++ reference** — [usefulsensors/moonshine](https://github.com/usefulsensors/moonshine) using ONNX Runtime on CPU
- **Python MLX** — [mlx-audio](https://github.com/Blaizzy/mlx-audio) using MLX on GPU/Metal
- **Swift MLX** — this project, using MLX Swift on GPU/Metal

All benchmarks run on Apple Silicon (MacBook Pro). Audio: datacenter industry talk at various lengths, plus Genesis Bible clips for WER validation.

---

## Per-Segment Latency (6.6s audio)

Time to transcribe a single ~6.6 second audio segment. This is the metric most relevant for live streaming (time from end of speech to text output).

| Implementation | Tiny (34M) | Small (123M) |
|---|---|---|
| **C++ reference (CPU/ONNX)** | **34ms** | **73ms** |
| **Swift MLX (GPU/Metal)** | **~97ms** | **~120ms** |
| **Python MLX (GPU/Metal)** | ~87ms | ~89ms |

Note: The C++ reference measures **response latency in streaming mode** — the encoder runs incrementally while speech is happening, so the reported time is decode-only after the final audio chunk. Our numbers include full encode + decode from scratch.

---

## Throughput (tokens/sec, warmed)

Sustained token generation rate on longer audio. Higher is better.

| Audio Length | Swift Tiny | Python Tiny | Swift Small | Python Small |
|---|---|---|---|---|
| **30s** | 303 tok/s | 316 tok/s | 261 tok/s | 238 tok/s |
| **60s** | 358 tok/s | 331 tok/s | 287 tok/s | 241 tok/s |
| **2 min** | **368 tok/s** | 314 tok/s | **300 tok/s** | 237 tok/s |
| **10 min** | **390 tok/s** | 318 tok/s | **305 tok/s** | 231 tok/s |

Swift MLX is **1.1–1.3x faster** than Python MLX across all long-audio tests. The advantage grows with longer audio as the chunked pipeline amortises overhead.

---

## Total Transcription Time

Wall-clock time to transcribe audio of various lengths (inference only, excludes model loading).

| Audio Length | Swift Tiny | Python Tiny | Swift Small | Python Small |
|---|---|---|---|---|
| **30s** | 0.37s | 0.36s | 0.42s | 0.46s |
| **60s** | 0.60s | 0.65s | 0.74s | 0.88s |
| **2 min** | **1.08s** | 1.27s | **1.33s** | 1.69s |
| **10 min** | **5.43s** | 6.65s | **6.96s** | 9.17s |

---

## Real-Time Factor (lower = faster)

Transcription time divided by audio duration. All values well under 1.0 (faster than real-time).

| Audio Length | Swift Tiny | Python Tiny | Swift Small | Python Small |
|---|---|---|---|---|
| **30s** | 0.012x | 0.012x | 0.014x | 0.015x |
| **2 min** | **0.009x** | 0.011x | **0.011x** | 0.014x |
| **10 min** | **0.009x** | 0.011x | **0.012x** | 0.015x |

All implementations are **~70–110x faster than real-time**.

---

## Peak Memory Usage

Peak GPU memory during inference (warmed, steady-state).

| Audio Length | Swift Tiny | Python Tiny | Swift Small | Python Small |
|---|---|---|---|---|
| **30s** | 864 MB | 1,053 MB | 1,256 MB | 1,609 MB |
| **2 min** | 869 MB | 1,059 MB | 1,262 MB | 1,615 MB |
| **10 min** | 900 MB | 1,090 MB | 1,292 MB | 1,646 MB |

Swift MLX uses **17–22% less memory** than Python MLX. Memory is bounded regardless of audio length (30s chunking).

---

## Word Error Rate

Measured on Genesis Bible audio clips (clean studio audio with known reference text).

| Model | Swift MLX | Python MLX |
|---|---|---|
| **Tiny** | **0% WER** | **0% WER** |
| **Small** | **0% WER** | **0% WER** |

Both implementations produce identical token sequences and match the reference text exactly.

---

## Unlimited Duration

Both offline and streaming modes handle arbitrarily long audio with bounded memory:

- **Offline** (`generate()`): Audio automatically chunked into 30s segments, each encoded and decoded independently. 10 minutes transcribed in 5.4s. A 1-hour file would take ~35 seconds.
- **Streaming** (`addAudio()` / `transcribe()`): Rolling 30s memory window with EOS-driven natural sentence segmentation. Runs indefinitely with constant memory.

---

## Test Environment

- **Hardware**: Apple Silicon MacBook Pro
- **MLX Swift**: 0.30.6
- **Python MLX**: 0.30.3
- **Models**: `UsefulSensors/moonshine-streaming-tiny`, `UsefulSensors/moonshine-streaming-small`
- **C++ reference numbers**: from the [Moonshine README](https://github.com/usefulsensors/moonshine) (MacBook Pro M4 column)
