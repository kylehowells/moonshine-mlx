import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Attention (uses MLXFast.RoPE traditional=true + scaledDotProductAttention)

final class MoonshineAttention: Module, @unchecked Sendable {
    let num_heads: Int
    let num_kv_heads: Int
    let head_dim: Int
    let num_kv_groups: Int
    let is_causal: Bool
    let scale: Float
    let use_rope: Bool
    let rotary_ndims: Int
    let ropeBase: Float

    let q_proj: Linear
    let k_proj: Linear
    let v_proj: Linear
    let o_proj: Linear

    init(
        inputDim: Int,
        numHeads: Int,
        headDim: Int,
        numKVHeads: Int,
        isCausal: Bool = false,
        useRope: Bool = false,
        partialRotaryFactor: Double = 0.8,
        ropeTheta: Double = 10000.0
    ) {
        self.num_heads = numHeads
        self.num_kv_heads = numKVHeads
        self.head_dim = headDim
        self.num_kv_groups = numHeads / numKVHeads
        self.is_causal = isCausal
        self.scale = pow(Float(headDim), -0.5)

        let attnDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        self.q_proj = Linear(inputDim, attnDim, bias: false)
        self.k_proj = Linear(inputDim, kvDim, bias: false)
        self.v_proj = Linear(inputDim, kvDim, bias: false)
        self.o_proj = Linear(attnDim, inputDim, bias: false)

        self.use_rope = useRope
        if useRope {
            var nd = Int(Double(headDim) * partialRotaryFactor)
            nd -= nd % 2
            self.rotary_ndims = nd
        } else {
            self.rotary_ndims = 0
        }
        self.ropeBase = Float(ropeTheta)
    }

    struct Output {
        var output: MLXArray
        var keyCache: MLXArray
        var valueCache: MLXArray
        var crossQK: MLXArray?
    }

    func callAsFunction(
        _ x: MLXArray,
        encoderHiddenStates: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil,
        mask: MLXArray? = nil,
        returnWeights: Bool = false
    ) -> Output {
        let B = x.dim(0)
        let T = x.dim(1)
        let isCross = encoderHiddenStates != nil

        // Q projection always needed
        var q = q_proj(x).reshaped(B, T, num_heads, head_dim).transposed(0, 2, 1, 3)

        var k: MLXArray
        var v: MLXArray

        if isCross, let cache {
            // Cross-attention with cached KV: skip K/V projection entirely
            k = cache.0
            v = cache.1
        } else {
            let kvInput = encoderHiddenStates ?? x
            let S = kvInput.dim(1)
            k = k_proj(kvInput).reshaped(B, S, num_kv_heads, head_dim).transposed(0, 2, 1, 3)
            v = v_proj(kvInput).reshaped(B, S, num_kv_heads, head_dim).transposed(0, 2, 1, 3)

            if use_rope && !isCross {
                let offset = cache?.0.dim(2) ?? 0
                q = MLXFast.RoPE(q, dimensions: rotary_ndims, traditional: true,
                                 base: ropeBase, scale: 1.0, offset: offset)
                k = MLXFast.RoPE(k, dimensions: rotary_ndims, traditional: true,
                                 base: ropeBase, scale: 1.0, offset: offset)
            }

            if let cache {
                // Self-attention: append to cache
                k = concatenated([cache.0, k], axis: 2)
                v = concatenated([cache.1, v], axis: 2)
            }
        }

        var kExpanded = k
        var vExpanded = v
        if num_kv_groups > 1 {
            kExpanded = MLX.repeated(k, count: num_kv_groups, axis: 1)
            vExpanded = MLX.repeated(v, count: num_kv_groups, axis: 1)
        }

        var attnMask = mask
        if is_causal && T > 1 {
            var causal = MultiHeadAttention.createAdditiveCausalMask(T)
            if causal.dtype != q.dtype { causal = causal.asType(q.dtype) }
            let kLen = kExpanded.dim(2)
            if kLen > T {
                let prefix = MLXArray.zeros([T, kLen - T]).asType(q.dtype)
                attnMask = concatenated([prefix, causal], axis: 1)
            } else {
                attnMask = causal
            }
        }

        var crossQK: MLXArray? = nil
        let o: MLXArray

        if returnWeights && isCross {
            let qk = q.matmul(kExpanded.transposed(0, 1, 3, 2)) * scale
            let w = softmax(qk, axis: -1)
            o = w.matmul(vExpanded)
            crossQK = qk
        } else {
            if let attnMask {
                o = MLXFast.scaledDotProductAttention(
                    queries: q, keys: kExpanded, values: vExpanded,
                    scale: scale, mask: attnMask
                )
            } else {
                // Manual attention for maskless paths (cross-attention).
                // MLXFast.scaledDotProductAttention without a mask has a kernel bug
                // producing NaN for sequences > ~1024.
                let qk = q.matmul(kExpanded.transposed(0, 1, 3, 2)) * scale
                o = softmax(qk, axis: -1).matmul(vExpanded)
            }
        }

        let out = o.transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return Output(output: o_proj(out), keyCache: k, valueCache: v, crossQK: crossQK)
    }
}

// MARK: - MLPs

final class EncoderMLP: Module, @unchecked Sendable {
    let fc1: Linear
    let fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self.fc1 = Linear(hiddenSize, intermediateSize, bias: true)
        self.fc2 = Linear(intermediateSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

final class DecoderMLP: Module, @unchecked Sendable {
    let fc1: Linear
    let fc2: Linear
    let intermediateSize: Int

    init(hiddenSize: Int, intermediateSize: Int) {
        self.intermediateSize = intermediateSize
        self.fc1 = Linear(hiddenSize, 2 * intermediateSize, bias: true)
        self.fc2 = Linear(intermediateSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = fc1(x)
        // Try fused SwiGLU kernel (fuses split + silu + multiply into one dispatch)
        if let fused = MoonshineKernelFusion.shared.fusedSwiGLUGate(
            projected: projected, intermediateSize: intermediateSize
        ) {
            return fc2(fused)
        }
        // Fallback
        let splits = split(projected, parts: 2, axis: -1)
        return fc2(silu(splits[1]) * splits[0])
    }
}

// MARK: - Layer Normalization

/// LayerNorm with unit offset: gamma stored near 0, 1.0 added at runtime.
/// Uses MLXFast.layerNorm fused kernel with pre-computed (weight+1).
final class UnitOffsetLayerNorm: Module, @unchecked Sendable {
    var weight: MLXArray
    let dims: Int
    /// Pre-computed and eval'd (weight + 1). Set by prepareForInference().
    var offsetWeight: MLXArray?

    init(_ dims: Int) {
        self.dims = dims
        self.weight = MLXArray.zeros([dims])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: offsetWeight ?? (weight + 1.0), bias: nil, eps: 1e-5)
    }

    func prepareForInference() {
        offsetWeight = weight + 1.0
        eval(offsetWeight!)
    }
}

// MARK: - Encoder Layer

final class EncoderLayer: Module, @unchecked Sendable {
    let self_attn: MoonshineAttention
    let mlp: EncoderMLP
    let input_layernorm: UnitOffsetLayerNorm
    let post_attention_layernorm: UnitOffsetLayerNorm

    let windowLeft: Int?
    let windowRight: Int?
    private var _cachedMask: MLXArray?
    private var _cachedMaskLen: Int = 0

    init(config: MoonshineModelConfig, layerIdx: Int) {
        let ec = config.encoder
        self.self_attn = MoonshineAttention(
            inputDim: ec.hiddenSize, numHeads: ec.numAttentionHeads,
            headDim: ec.headDim, numKVHeads: ec.numKeyValueHeads,
            isCausal: false, useRope: false
        )
        self.mlp = EncoderMLP(hiddenSize: ec.hiddenSize, intermediateSize: ec.intermediateSize)
        self.input_layernorm = UnitOffsetLayerNorm(ec.hiddenSize)
        self.post_attention_layernorm = UnitOffsetLayerNorm(ec.hiddenSize)

        if let windows = ec.slidingWindows, layerIdx < windows.count {
            self.windowLeft = windows[layerIdx][0]
            self.windowRight = windows[layerIdx][1]
        } else {
            self.windowLeft = nil
            self.windowRight = nil
        }
    }

    private func slidingMask(seqLen: Int) -> MLXArray? {
        guard let windowLeft, let windowRight else { return nil }
        if seqLen == _cachedMaskLen, let cached = _cachedMask { return cached }
        // Build mask with explicit Float computation to avoid type issues
        let pos = MLXArray(0 ..< Int32(seqLen))
        let qPos = expandedDimensions(pos, axis: 1)  // [seqLen, 1]
        let kPos = expandedDimensions(pos, axis: 0)  // [1, seqLen]
        let diff = qPos - kPos  // [seqLen, seqLen] int32

        // For each (q, k): valid if (q-k) in [0, windowLeft) OR (k-q) in (0, windowRight)
        var mask: MLXArray
        if windowRight > 0 {
            let leftOK = (diff .>= 0) .&& (diff .< Int32(windowLeft))
            let rightOK = (diff .< 0) .&& ((-diff) .< Int32(windowRight))
            mask = which(leftOK .|| rightOK, MLXArray(Float(0.0)), MLXArray(Float(-1e9)))
        } else {
            // No right context - simpler mask
            let valid = (diff .>= 0) .&& (diff .< Int32(windowLeft))
            mask = which(valid, MLXArray(Float(0.0)), MLXArray(Float(-1e9)))
        }
        eval(mask)  // Force materialization to avoid lazy graph issues
        _cachedMask = mask
        _cachedMaskLen = seqLen
        return mask
    }

    func clearMaskCache() {
        _cachedMask = nil
        _cachedMaskLen = 0
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var mask = slidingMask(seqLen: x.dim(1))
        // Ensure mask dtype matches input for scaledDotProductAttention
        if let m = mask, m.dtype != x.dtype {
            mask = m.asType(x.dtype)
        }
        var out = x
        let normed1 = input_layernorm(out)
        let attnResult = self_attn(normed1, mask: mask)
        out = out + attnResult.output
        let normed2 = post_attention_layernorm(out)
        out = out + mlp(normed2)
        return out
    }
}

// MARK: - Decoder Layer

final class DecoderLayer: Module, @unchecked Sendable {
    let self_attn: MoonshineAttention
    let encoder_attn: MoonshineAttention
    let mlp: DecoderMLP
    let input_layernorm: LayerNorm
    let post_attention_layernorm: LayerNorm
    let final_layernorm: LayerNorm

    init(config: MoonshineModelConfig) {
        let d = config.hiddenSize
        self.self_attn = MoonshineAttention(
            inputDim: d, numHeads: config.numAttentionHeads,
            headDim: config.headDim, numKVHeads: config.numKeyValueHeads,
            isCausal: true, useRope: true,
            partialRotaryFactor: config.partialRotaryFactor,
            ropeTheta: config.ropeTheta
        )
        self.encoder_attn = MoonshineAttention(
            inputDim: d, numHeads: config.numAttentionHeads,
            headDim: config.headDim, numKVHeads: config.numKeyValueHeads,
            isCausal: false, useRope: false
        )
        self.mlp = DecoderMLP(hiddenSize: d, intermediateSize: config.intermediateSize)
        self.input_layernorm = LayerNorm(dimensions: d, bias: false)
        self.post_attention_layernorm = LayerNorm(dimensions: d, bias: false)
        self.final_layernorm = LayerNorm(dimensions: d, bias: false)
    }

    struct Output {
        var hidden: MLXArray
        var selfCache: (MLXArray, MLXArray)
        var crossCache: (MLXArray, MLXArray)
        var crossQK: MLXArray?
    }

    func callAsFunction(
        _ x: MLXArray,
        memory: MLXArray,
        selfCache: (MLXArray, MLXArray)? = nil,
        crossCache: (MLXArray, MLXArray)? = nil,
        returnCrossWeights: Bool = false
    ) -> Output {
        var out = x

        let normed1 = input_layernorm(out)
        let selfResult = self_attn(normed1, cache: selfCache)
        out = out + selfResult.output

        let normed2 = post_attention_layernorm(out)
        let crossResult = encoder_attn(
            normed2, encoderHiddenStates: memory,
            cache: crossCache, returnWeights: returnCrossWeights
        )
        out = out + crossResult.output

        let normed3 = final_layernorm(out)
        out = out + mlp(normed3)

        return Output(
            hidden: out,
            selfCache: (selfResult.keyCache, selfResult.valueCache),
            crossCache: (crossResult.keyCache, crossResult.valueCache),
            crossQK: crossResult.crossQK
        )
    }
}
