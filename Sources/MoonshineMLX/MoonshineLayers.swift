import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Rotary Embeddings (decoder self-attention)

final class RotaryEmbedding: Module, @unchecked Sendable {
    let dim: Int
    let base: Float

    init(dim: Int, base: Float = 10000.0) {
        self.dim = dim
        self.base = base
    }

    func callAsFunction(positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let invFreq = 1.0 / MLX.pow(
            MLXArray(base),
            MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        )
        // positionIds: [B, T] -> [B, T, 1]
        let ids = positionIds.expandedDimensions(axis: -1).asType(.float32)
        // invFreq: [dim/2] -> [1, 1, dim/2]
        let freq = invFreq.reshaped(1, 1, -1)
        let freqs = ids * freq  // [B, T, dim/2]
        let emb = concatenated([freqs, freqs], axis: -1)  // [B, T, dim]
        return (MLX.cos(emb), MLX.sin(emb))
    }
}

func rotateHalf(_ x: MLXArray) -> MLXArray {
    let x1 = x[.ellipsis, .stride(from: 0, by: 2)]
    let x2 = x[.ellipsis, .stride(from: 1, by: 2)]
    return stacked([-x2, x1], axis: -1).reshaped(x.shape)
}

func applyRotaryPosEmb(
    q: MLXArray, k: MLXArray,
    cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    // cos, sin: [B, T, D] -> [B, 1, T, D] for broadcasting with [B, H, T, D]
    var c = cos.expandedDimensions(axis: 1)
    var s = sin.expandedDimensions(axis: 1)
    let half = c.dim(-1) / 2
    c = MLX.repeated(c[.ellipsis, ..<half], count: 2, axis: -1)
    s = MLX.repeated(s[.ellipsis, ..<half], count: 2, axis: -1)

    let rotDim = c.dim(-1)

    let qRot = q[.ellipsis, ..<rotDim]
    let qPass = q[.ellipsis, rotDim...]
    let kRot = k[.ellipsis, ..<rotDim]
    let kPass = k[.ellipsis, rotDim...]

    let qOut = qRot * c + rotateHalf(qRot) * s
    let kOut = kRot * c + rotateHalf(kRot) * s

    return (
        concatenated([qOut, qPass], axis: -1),
        concatenated([kOut, kPass], axis: -1)
    )
}

// MARK: - Attention

final class MoonshineAttention: Module, @unchecked Sendable {
    let num_heads: Int
    let num_kv_heads: Int
    let head_dim: Int
    let num_kv_groups: Int
    let is_causal: Bool
    let scale: Float
    let use_rope: Bool
    let rotary_ndims: Int

    let q_proj: Linear
    let k_proj: Linear
    let v_proj: Linear
    let o_proj: Linear

    var rotary_emb: RotaryEmbedding?

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
            var rotNdims = Int(Double(headDim) * partialRotaryFactor)
            rotNdims -= rotNdims % 2
            self.rotary_ndims = rotNdims
            self.rotary_emb = RotaryEmbedding(dim: rotNdims, base: Float(ropeTheta))
        } else {
            self.rotary_ndims = 0
            self.rotary_emb = nil
        }
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
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        returnWeights: Bool = false
    ) -> Output {
        let B = x.dim(0)
        let T = x.dim(1)
        let isCross = encoderHiddenStates != nil

        var q = q_proj(x)
        let kvInput = encoderHiddenStates ?? x
        var k = k_proj(kvInput)
        var v = v_proj(kvInput)

        q = q.reshaped(B, T, num_heads, head_dim).transposed(0, 2, 1, 3)
        let S = k.dim(1)
        k = k.reshaped(B, S, num_kv_heads, head_dim).transposed(0, 2, 1, 3)
        v = v.reshaped(B, S, num_kv_heads, head_dim).transposed(0, 2, 1, 3)

        if use_rope && !isCross {
            let ids: MLXArray
            if let positionIds {
                ids = positionIds
            } else {
                let offset = cache?.0.dim(2) ?? 0
                ids = MLXArray(Int32(offset) ..< Int32(offset + T)).reshaped(1, T)
            }
            let (cos, sin) = rotary_emb!(positionIds: ids)
            (q, k) = applyRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)
        }

        if let cache {
            if isCross {
                k = cache.0
                v = cache.1
            } else {
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
            let causal = MultiHeadAttention.createAdditiveCausalMask(T)
            let kLen = kExpanded.dim(2)
            var causalFull: MLXArray
            if kLen > T {
                let prefix = MLXArray.zeros([T, kLen - T])
                causalFull = concatenated([prefix, causal], axis: 1)
            } else {
                causalFull = causal
            }
            if let existing = attnMask {
                attnMask = existing + causalFull
            } else {
                attnMask = causalFull
            }
        }

        var crossQK: MLXArray? = nil
        var o: MLXArray

        if returnWeights && isCross {
            // Compute attention weights explicitly for timestamp extraction
            let qk = (q.matmul(kExpanded.transposed(0, 1, 3, 2))) * MLXArray(scale)
            let w = softmax(qk, axis: -1)
            o = w.matmul(vExpanded)
            crossQK = qk  // [B, H, T, S] raw scores
        } else {
            o = MLXFast.scaledDotProductAttention(
                queries: q, keys: kExpanded, values: vExpanded,
                scale: scale, mask: attnMask
            )
        }

        o = o.transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return Output(output: o_proj(o), keyCache: k, valueCache: v, crossQK: crossQK)
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
    /// SwiGLU: fc1 -> split(x, gate) -> silu(gate) * x -> fc2
    let fc1: Linear
    let fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self.fc1 = Linear(hiddenSize, 2 * intermediateSize, bias: true)
        self.fc2 = Linear(intermediateSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = fc1(x)
        let splits = split(projected, parts: 2, axis: -1)
        return fc2(silu(splits[1]) * splits[0])
    }
}

// MARK: - Layer Normalization

/// LayerNorm with unit offset: gamma is stored near 0, and 1.0 is added at runtime.
/// Matches HuggingFace MoonshineStreamingLayerNorm (weight key: *.gamma -> *.weight after sanitize).
final class UnitOffsetLayerNorm: Module, @unchecked Sendable {
    var weight: MLXArray
    let dims: Int

    init(_ dims: Int) {
        self.dims = dims
        self.weight = MLXArray.zeros([dims])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        let normed = (x - mean) / MLX.sqrt(variance + 1e-5)
        return normed * (weight + 1.0)
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
        let pos = MLXArray(0 ..< Int32(seqLen))
        let diff = expandedDimensions(pos, axis: 1) - expandedDimensions(pos, axis: 0)
        let wl = MLXArray(Int32(windowLeft))
        let wr = MLXArray(Int32(windowRight))
        let leftValid = (diff .>= 0) .&& (diff .< wl)
        let rightValid = (diff .< 0) .&& ((-diff) .< wr)
        let valid = leftValid .|| rightValid
        return which(valid, MLXArray(Float(0.0)), MLXArray(Float(-1e9)))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mask = slidingMask(seqLen: x.dim(1))
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

        // Self-attention
        let normed1 = input_layernorm(out)
        let selfResult = self_attn(normed1, cache: selfCache)
        out = out + selfResult.output

        // Cross-attention
        let normed2 = post_attention_layernorm(out)
        let crossResult = encoder_attn(
            normed2, encoderHiddenStates: memory,
            cache: crossCache, returnWeights: returnCrossWeights
        )
        out = out + crossResult.output

        // FFN
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
