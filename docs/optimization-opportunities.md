# Remaining Optimization Opportunities

Current performance baseline (tiny model, 30s audio, 111 tokens, Apple Silicon):
- **Per-token decode: 3.46ms** (~95 GPU dispatches per token)
- **Total: 416ms** (91% spent in autoregressive decode loop)
- **Peak memory: ~864 MB** (176 MB weights + cache + intermediates)

---

## Speed

### 1. Reduce GPU Dispatch Count

**Impact: High — could halve per-token latency**

Each decode step launches ~95 separate GPU kernels (layernorms, matmuls, SDPA, residual adds, etc). At ~0.03ms dispatch overhead each, this accounts for ~2.9ms of the 3.46ms per-token cost. The actual compute is only ~0.5ms.

Options:
- **Fused decoder layer kernel**: A custom Metal compute pipeline that fuses an entire decoder layer (norm → self-attn → norm → cross-attn → norm → FFN) into ~3 dispatches instead of ~15. This is what the DeepFilterNet-mlx project does for its streaming layers.
- **MLX `compile()`**: Should fuse operations automatically, but currently doesn't work with the autoregressive decode loop because KV cache shapes change each step. Could work if the cache were pre-allocated to max size.
- **Speculative decoding**: Generate multiple candidate tokens per step to amortise dispatch overhead across more useful work.

### 2. Fix the SDPA Maskless Kernel Bug

**Impact: Medium — ~0.3ms/token saving**

`MLXFast.scaledDotProductAttention` produces NaN when called without a mask for sequences > ~1024 (tracked in `bugs/sdpa-nan-maskless.md`). We work around this by passing a `[T,S]` zeros mask, which forces a slower kernel path.

Once mlx-swift fixes this (the Python MLX equivalent works correctly), removing the zeros mask would use the fastest fused attention path for cross-attention. This saves one allocation + one addition per cross-attention per layer per token = 12 fewer ops per step.

### 3. Streaming Encode Pipelining

**Impact: Medium — reduces perceived latency for live use**

The C++ reference encodes audio incrementally *while speech is happening*, so by the time the user finishes speaking, most of the encoding is already done. Their reported 34ms latency is decode-only.

Our streaming `transcribe()` already supports incremental encoding, but calling it more frequently during `addAudio()` (e.g., every 100ms of new audio) would overlap encode work with speech input, reducing end-of-utterance latency from ~97ms toward the C++ reference's 34ms.

### 4. Compile the Encoder

**Impact: Medium — could reduce encode time from 25ms to ~10ms**

The encoder has fixed shapes per audio chunk (determined by audio length). This makes it a good candidate for `compile(shapeless: false)`, which would fuse the 6 transformer layers into fewer GPU dispatches.

The encoder is only 6% of total time for 30s audio, but for short segments (5-10s in streaming mode) it becomes a larger fraction of latency.

---

## Memory

### 5. Float16 Weights

**Impact: High — would halve weight memory (176MB → 88MB tiny, 561MB → 280MB small)**

Full float16 broke output quality in our testing. However, selective float16 may work:
- Keep encoder normalization layers and embedding tables in float32
- Cast decoder `Linear` weight matrices to float16 (these are the bulk of the parameters)
- The `scaledDotProductAttention` kernel already accumulates in higher precision internally

This needs careful quality validation (WER testing across diverse audio).

### 6. Quantization (4-bit / 8-bit)

**Impact: High — 4x–2x memory reduction**

MLX Swift supports `QuantizedLinear` with per-group int4/int8 scales. This would reduce the tiny model from 176MB to ~44MB (4-bit) or ~88MB (8-bit).

Blockers:
- No pre-quantized Moonshine V2 weights on HuggingFace yet
- Could quantize at load time using `MLXNN.quantize()`, but quality impact needs measurement
- The Python mlx-audio has a conversion script (`convert.py`) that supports quantization — could be adapted

### 7. Encoder Peak Memory

**Impact: Medium — reduce 1,006MB encode peak**

The sliding window attention masks are materialised as `[seqLen, seqLen]` float32 matrices. For 1500 frames (30s), that's 9MB per layer × 6 layers = 54MB just for masks.

Options:
- Generate masks on-the-fly inside the SDPA kernel (requires MLX support for custom attention patterns)
- Process the encoder in temporal chunks to reduce peak intermediate activation memory
- Use sparse attention masks instead of dense

### 8. proj_out Weight Sharing

**Impact: Low — would save 42MB (tiny) or 67MB (small)**

The `proj_out` layer ([vocab_size × hidden_size]) could share weights with the embedding table if `tie_word_embeddings` were enabled. However, the HuggingFace model config has `tie_word_embeddings: false`, so this is a model-level choice that can't be changed without retraining.

---

## Quality / Features

### 9. Voice Activity Detection (VAD)

**Impact: High for streaming quality**

The C++ reference uses Silero VAD to detect speech boundaries (pauses, silence) and segment audio at natural points. This eliminates the boundary artifacts that occur when segments are forced at 30s.

Options:
- Port Silero VAD v5 to MLX Swift (~2MB model)
- Use Apple's built-in speech detection via AVAudioEngine
- Simple energy-based VAD as a lightweight alternative

### 10. Overlap-Merge for Offline Chunking

**Impact: Medium for offline quality**

Current 30s chunking has hard cuts between segments that can split sentences. Processing with 5s overlap and merging using word timestamps would eliminate boundary artifacts:

1. Chunk audio into 30s segments with 5s overlap
2. Transcribe each segment with word-level timestamps
3. For overlapping regions, use timestamps to deduplicate and select the highest-confidence words

Adds ~17% compute but produces seamless text across segment boundaries.
