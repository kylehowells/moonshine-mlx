# moonshine-mlx

Native Swift MLX/Metal implementation of [Moonshine V2 Streaming](https://github.com/usefulsensors/moonshine) speech-to-text for Apple Silicon (macOS & iOS).

Built on [MLX Swift](https://github.com/ml-explore/mlx-swift), this library runs Moonshine's encoder-decoder transformer entirely on the GPU via Metal, with no Python dependencies.

## Features

- **Offline transcription** - single-pass encode + autoregressive decode
- **Streaming transcription** - chunked audio with incremental encoder/decoder and causal convolution buffers
- **Word-level timestamps** - cross-attention DTW alignment
- **HuggingFace Hub integration** - automatic model download from [UsefulSensors/moonshine-streaming-tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) and [small](https://huggingface.co/UsefulSensors/moonshine-streaming-small)
- **Custom Metal kernels** - fused CMVN+asinh frontend, fused SwiGLU gate
- **Optimised decoding** - MLXFast RoPE, fused layer normalization, cross-attention KV caching, async eval pipelining

## Performance

Benchmarked against the [Python MLX reference](https://github.com/Blaizzy/mlx-audio) (mlx-audio) on the same hardware and models:

| Model | Audio | Swift | Python | Speedup |
|-------|-------|-------|--------|---------|
| tiny | 6.6s | 0.063s (291 tok/s) | 0.073s (250 tok/s) | **1.16x** |
| tiny | 13.2s | 0.111s (344 tok/s) | 0.131s (296 tok/s) | **1.18x** |
| small | 13.2s | 0.133s (285 tok/s) | 0.168s (227 tok/s) | **1.26x** |

- **0% WER** on all test files (identical output to Python)
- **15% less memory** than Python (265 MB vs 584 MB for tiny, 797 MB vs 931 MB for small)
- **2.1ms per token** on 30s audio (tiny model, warmed)

## Usage

### Swift Package

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/kylehowells/moonshine-mlx.git", branch: "master"),
]
```

```swift
import MoonshineMLX

let model = try MoonshineModel.load(from: "UsefulSensors/moonshine-streaming-tiny")
let (_, audio) = try AudioIO.loadMono(url: audioURL, targetSampleRate: model.sampleRate)
let result = model.generate(audio: audio)
print(result.text)
```

### CLI

```bash
swift build -c release
.build/release/moonshine-mlx audio.wav --model UsefulSensors/moonshine-streaming-tiny --verbose
```

Options:
- `--stream` - streaming mode (process audio in chunks)
- `--word-timestamps` - word-level timing
- `--json` - JSON metrics output
- `--profile` - encode/decode timing breakdown

## Models

| Model | Params | HuggingFace |
|-------|--------|-------------|
| Tiny | 43M | [UsefulSensors/moonshine-streaming-tiny](https://huggingface.co/UsefulSensors/moonshine-streaming-tiny) |
| Small | 147M | [UsefulSensors/moonshine-streaming-small](https://huggingface.co/UsefulSensors/moonshine-streaming-small) |
| Medium | 245M | [UsefulSensors/moonshine-streaming-medium](https://huggingface.co/UsefulSensors/moonshine-streaming-medium) |

## Requirements

- macOS 14+ or iOS 17+
- Apple Silicon (M1/M2/M3/M4)
- Swift 6.2+

## References

- **Moonshine** - [github.com/usefulsensors/moonshine](https://github.com/usefulsensors/moonshine)
- **Moonshine V2 paper** - [Moonshine v2: Ergodic Streaming Encoder ASR for Latency-Critical Speech Applications](https://arxiv.org/abs/2602.12241)
- **Moonshine V1 paper** - [Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) (Jeffries et al., 2024)
- **MLX Swift** - [github.com/ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift)
- **mlx-audio** (Python reference) - [github.com/Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)

## License

MIT
