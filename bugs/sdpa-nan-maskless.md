# MLXFast.scaledDotProductAttention NaN without mask (FIXED)

> **Status: Fixed.** MLX C++ commit `99ca62c4` ("Fix 2pass sdpa on < M2") resolves this.
> The fix is included in mlx-swift 0.30.6's MLX C++ submodule. If you still see this
> issue, ensure you've done a clean rebuild (`swift package clean && swift build -c release`)
> so the metallib is recompiled with the fix.

## Summary

`MLXFast.scaledDotProductAttention` produces NaN output when called **without a mask** and the key/value sequence length exceeds approximately 1024. The same operation with an explicit mask (even a mask of all zeros) does **not** produce NaN. The equivalent Python `mx.fast.scaled_dot_product_attention` works correctly at all sizes.

## Affected Version

- **mlx-swift**: 0.30.6
- **Platform**: macOS 15, Apple M-series (tested on Apple Silicon)
- **Python mlx**: 0.30.3 — **not affected**

## Reproduction

The bug manifests in cross-attention where queries have T=1 and keys/values have S > ~1024, with no attention mask.

### Minimal Swift Reproducer

Add this as an executable target in a package that depends on `mlx-swift` 0.30.6:

```swift
import MLX
import MLXFast
import MLXRandom

// Test SDPA with no mask at various key sequence lengths
for S in [512, 1000, 1024, 1025, 1050, 2048] {
    let q = MLXRandom.normal([1, 8, 1, 64])   // [B, H, T=1, D]
    let k = MLXRandom.normal([1, 8, S, 64])   // [B, H, S,   D]
    let v = MLXRandom.normal([1, 8, S, 64])
    eval(q, k, v)

    // Without mask — produces NaN for S > ~1024
    let out = MLXFast.scaledDotProductAttention(
        queries: q, keys: k, values: v,
        scale: 0.125, mask: nil
    )
    eval(out)
    let hasNaN = MLX.any(MLX.isNaN(out)).item(Bool.self)
    print("S=\(S) mask=nil  -> NaN=\(hasNaN)")

    // Manual attention — always correct
    let qk = q.matmul(k.transposed(0, 1, 3, 2)) * 0.125
    let manual = softmax(qk, axis: -1).matmul(v)
    eval(manual)
    let manualNaN = MLX.any(MLX.isNaN(manual)).item(Bool.self)
    print("S=\(S) manual   -> NaN=\(manualNaN)")
}
```

### Expected Output

All lines should show `NaN=false`.

### Actual Output

```
S=512  mask=nil  -> NaN=false
S=512  manual   -> NaN=false
S=1000 mask=nil  -> NaN=false
S=1000 manual   -> NaN=false
S=1024 mask=nil  -> NaN=false
S=1024 manual   -> NaN=false
S=1025 mask=nil  -> NaN=true     <-- BUG
S=1025 manual   -> NaN=false
S=1050 mask=nil  -> NaN=true     <-- BUG
S=1050 manual   -> NaN=false
S=2048 mask=nil  -> NaN=true     <-- BUG
S=2048 manual   -> NaN=false
```

*(Exact threshold may vary by hardware and head dimensions. Observed at S=1050 with H=8, D=64 on M-series. Models with D=40 may have a higher threshold.)*

### Python Equivalent (Works Correctly)

```python
import mlx.core as mx

for S in [512, 1000, 1024, 1025, 1050, 2048]:
    q = mx.random.normal([1, 8, 1, 64])
    k = mx.random.normal([1, 8, S, 64])
    v = mx.random.normal([1, 8, S, 64])
    mx.eval(q, k, v)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    mx.eval(out)
    print(f"S={S} nan={bool(mx.any(mx.isnan(out)))}")
# All print nan=False
```

## Impact

This affects any encoder-decoder transformer model using cross-attention without a mask when the encoder output exceeds ~1024 tokens. In practice this means:

- Speech-to-text models (Moonshine, Whisper) fail on audio longer than ~20 seconds
- Machine translation with long source sequences
- Any cross-attention over long sequences

## Workaround

Use manual attention for maskless calls and fused SDPA only when a mask is provided:

```swift
if let mask = attnMask {
    // Fused SDPA with mask — works correctly
    o = MLXFast.scaledDotProductAttention(
        queries: q, keys: k, values: v,
        scale: scale, mask: mask
    )
} else {
    // Manual attention — avoids the NaN bug
    let qk = q.matmul(k.transposed(0, 1, 3, 2)) * scale
    o = softmax(qk, axis: -1).matmul(v)
}
```

Performance cost: ~3 extra GPU kernel dispatches per attention layer per decode step (matmul, softmax, matmul vs 1 fused SDPA dispatch). For a 6-layer decoder this adds ~0.6ms per token.

## Notes

- SDPA **with** a mask works correctly at all sizes (encoder self-attention with sliding window masks tested up to 6000+ frames)
- Passing a zeros mask does **not** fix the issue — the output is still incorrect
- The bug appears to be in the Metal kernel dispatch path for the maskless case in mlx-swift specifically, not in the underlying MLX C++ library
