# Optimization Opportunities

## Current Performance

Baseline (Apple Silicon, warmed, 30s audio):

| Model | Tok/s | Warm 6s | 10min | Memory | WER |
|-------|-------|---------|-------|--------|-----|
| tiny-fp32 | 167 | 110ms | 5.33s | 256 MB | 2.8% |
| tiny-8bit | 156 | 120ms | 5.42s | 137 MB | 2.8% |
| small-8bit | 172 | 100ms | 6.33s | 599 MB | 1.9% |
| medium-4bit | 138 | 140ms | 10.44s | 453 MB | 2.4% |

---

## Why We're Near the Performance Floor

### The memory bandwidth bottleneck

Each decode token requires reading the model weights from GPU memory to compute the output. For the tiny model (34M parameters, float32):

- **Weight reads per token**: ~176 MB (every Linear layer weight is read once)
- **Apple M-series memory bandwidth**: ~100-200 GB/s (depending on chip)
- **Theoretical minimum time**: 176 MB ÷ 200 GB/s = **0.88ms per token**
- **Our actual time**: ~2.1ms per token (tiny fp32, warmed)

We're within ~2.4x of the memory bandwidth floor. The gap comes from:
- KV cache reads (grows with sequence length)
- Intermediate activations
- GPU dispatch overhead (~0.5ms total, hidden by async pipeline)
- Non-matmul operations (layernorm, softmax, RoPE, argmax)

For quantized models the gap closes further because weight reads are smaller:
- **tiny-8bit**: 88 MB weights → theoretical floor ~0.44ms → actual ~2.3ms (~5x gap, but cache dominates)
- **medium-fp16**: 532 MB weights → theoretical floor ~2.66ms → actual ~7.1ms (~2.7x gap)

The larger the model, the closer we are to the bandwidth floor because weight reads dominate over fixed overhead.

### Why custom Metal kernels didn't help

We tried fusing operations to reduce GPU dispatch count:

| Approach | Result | Why |
|---|---|---|
| `compile()` on encoder | No gain | Sliding window masks differ per sequence length → recompilation |
| `compile()` on decode step | **2.5x slower** | KV cache grows by 1 each step → recompilation every token |
| Fused residual+layernorm Metal kernel | No gain | Async eval pipeline already hides dispatch overhead |
| Fused SwiGLU Metal kernel | No gain | Same — pipeline hides it |

The key insight: MLX's **async eval pipeline** effectively overlaps GPU dispatch overhead with computation. The ~95 dispatches per token don't stall sequentially — while the GPU executes dispatch N, the CPU is already building dispatch N+1. The pipeline turns the serial dispatch overhead into a small latency addition rather than a throughput bottleneck.

This means fusing dispatches saves CPU-side graph-building time but not GPU execution time. Since the GPU is the bottleneck (memory bandwidth limited), reducing dispatches doesn't help.

---

## What's Already Implemented

- ✅ **MLXFast.RoPE** (fused rotary embeddings — `traditional=true`)
- ✅ **MLXFast.layerNorm** (fused layer normalization with pre-computed offset weights)
- ✅ **MLXFast.scaledDotProductAttention** (fused attention for masked paths)
- ✅ **Cross-attention KV skip** (skip K/V projection on cached decode steps)
- ✅ **Async eval pipelining** (overlap GPU compute with CPU graph building)
- ✅ **Attention mask caching** (avoid recomputing sliding window masks)
- ✅ **fp16, 8-bit, 4-bit quantized model support**
- ✅ **30s auto-chunking** for unlimited audio length

---

## Remaining Opportunities

### Speed

#### 1. Fix the SDPA Maskless Kernel Bug

**Impact: Low-Medium (~0.3ms/token)**

`MLXFast.scaledDotProductAttention` produces NaN without a mask for sequences > ~1024 in mlx-swift 0.30.6 (tracked in `bugs/sdpa-nan-maskless.md`). We work around this by passing a `[T,S]` zeros mask. Once mlx-swift fixes this, we can use the fastest kernel path for cross-attention.

#### 2. Streaming Encode Pipelining

**Impact: Medium for perceived latency**

The C++ reference encodes audio incrementally while speech is happening, so by the time the user finishes speaking most encoding is done. Their 34ms latency is decode-only.

Our streaming `transcribe()` supports incremental encoding but calling it more frequently during `addAudio()` would overlap encode with speech input, reducing end-of-utterance latency.

#### 3. Speculative Decoding

**Impact: Potentially High (2-3x decode throughput)**

Generate multiple candidate tokens per step using a smaller draft model, then verify in parallel with the full model. This amortises the per-step overhead across multiple tokens. Complex to implement but could break through the current bandwidth floor by increasing useful work per memory read.

### Memory

#### 4. Encoder Peak Memory

**Impact: Medium — reduce peak from ~1GB to ~500MB**

Sliding window attention masks materialised as `[seqLen, seqLen]` float32 matrices use 9MB per layer × 6 layers = 54MB for 30s audio. Options:
- Generate masks on-the-fly in SDPA (needs MLX support)
- Sparse attention masks instead of dense
- Process encoder in temporal chunks

### Quality

#### 5. Voice Activity Detection (VAD)

**Impact: High for streaming quality**

The C++ reference uses Silero VAD to segment at natural speech pauses. This eliminates boundary artifacts when segments are forced at 30s. Options:
- Port Silero VAD v5 to MLX Swift (~2MB model)
- Apple's built-in speech detection via AVAudioEngine
- Simple energy-based VAD as lightweight alternative

#### 6. Overlap-Merge for Offline Chunking

**Impact: Medium for offline quality**

Current 30s chunking can split sentences at boundaries. Processing with 5s overlap and merging using word timestamps would eliminate artifacts:
1. Chunk with 5s overlap
2. Transcribe each with word-level timestamps
3. Deduplicate overlap region using highest-confidence words

Adds ~17% compute but produces seamless text.
