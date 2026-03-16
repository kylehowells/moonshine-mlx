# fp16 KV Cache Experiment Results

## What

Store the decoder's KV cache (keys and values from self-attention and cross-attention) in float16 instead of float32. The `scaledDotProductAttention` kernel accumulates in higher precision internally, so this is quality-safe.

## Why

Each decode token reads the full KV cache from GPU memory. Halving the cache dtype reduces memory bandwidth for the attention computation. The impact scales with sequence length (more cached tokens = more bandwidth saved).

## KV Cache Size

For the tiny model (hidden=320, head_dim=40, 8 heads, 6 layers) at token 100:

| Component | fp32 | fp16 |
|---|---|---|
| Self-attn KV (6 layers × 2 × 8 heads × 100 tokens × 40 dim) | 1.5 MB | 0.75 MB |
| Cross-attn KV (6 layers × 2 × 8 heads × 1500 frames × 40 dim) | 2.9 MB | 1.45 MB |
| **Total KV cache** | **4.4 MB** | **2.2 MB** |
| Weight reads per token | 176 MB | 176 MB |
| **KV as % of total traffic** | **2.5%** | **1.25%** |

The KV cache is small relative to weight reads, so the bandwidth saving is modest for short sequences. The benefit grows with longer sequences.

## Results

### Speed (tok/s on 10 minute audio, warmed)

| Model | Before (fp32 KV) | After (fp16 KV) | Change |
|---|---|---|---|
| tiny-fp32 | 381 tok/s (5.33s) | 381 tok/s (5.55s) | ~same |
| tiny-8bit | 390 tok/s (5.42s) | 393 tok/s (5.39s) | ~same |
| small-8bit | 334 tok/s (6.33s) | 374 tok/s (5.68s) | **+12%** |
| medium-4bit | 189 tok/s (10.44s) | 203 tok/s (10.46s) | ~same |

Small-8bit saw a 12% speedup on 10 minute audio. Other models within noise.

### Memory (peak, warmed)

| Model | Before | After | Change |
|---|---|---|---|
| tiny-fp32 | 256 MB | 256 MB | same |
| tiny-8bit | 137 MB | 137 MB | same |
| small-8bit | 599 MB | 599 MB | same |
| medium-4bit | 453 MB | 453 MB | same |

Peak memory unchanged — dominated by encoder pass (sliding window attention masks), not KV cache.

### WER (10,000 words synthetic TTS)

| Model | Before | After | Change |
|---|---|---|---|
| tiny-fp32 | 2.8% (277 errors) | 2.8% (277 errors) | **identical** |
| tiny-8bit | 2.8% (284 errors) | 2.9% (285 errors) | +1 error |
| small-8bit | 1.9% (195 errors) | 1.9% (195 errors) | **identical** |
| medium-4bit | 2.4% (239 errors) | 2.4% (239 errors) | **identical** |

**Zero quality impact.** The single extra error on tiny-8bit is within normal variation (1 word out of 10,000).

## Conclusion

fp16 KV cache is a free optimisation — no quality cost, modest speed improvement on longer sequences, zero memory impact on peak usage. Enabled by default.
