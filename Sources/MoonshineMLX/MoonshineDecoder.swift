import Foundation
import MLX
import MLXNN

// MARK: - Decoder

final class MoonshineDecoder: Module, @unchecked Sendable {
    let embed_tokens: Embedding
    let pos_emb: Embedding
    let proj: Linear?
    let layers: [DecoderLayer]
    let norm: LayerNorm

    let encoderDim: Int
    let decoderDim: Int

    init(config: MoonshineModelConfig) {
        let encDim = config.encoderHiddenSize
        let decDim = config.hiddenSize
        self.encoderDim = encDim
        self.decoderDim = decDim

        self.embed_tokens = Embedding(embeddingCount: config.vocabSize, dimensions: decDim)
        self.pos_emb = Embedding(embeddingCount: config.maxPositionEmbeddings, dimensions: encDim)

        if encDim != decDim {
            self.proj = Linear(encDim, decDim, bias: false)
        } else {
            self.proj = nil
        }

        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            DecoderLayer(config: config)
        }
        self.norm = LayerNorm(dimensions: decDim, bias: false)
    }

    /// Add learned positional embedding and project to decoder dim.
    func prepareMemory(_ encoderOut: MLXArray, posOffset: Int = 0) -> MLXArray {
        let T = encoderOut.dim(1)
        let positions = MLXArray(Int32(posOffset) ..< Int32(posOffset + T))
        var x = encoderOut + pos_emb(positions)
        if let proj {
            x = proj(x)
        }
        return x
    }

    struct Output {
        var hidden: MLXArray
        var cache: [DecoderLayerCache]
        var crossQKAll: [[MLXArray]]?
    }

    func callAsFunction(
        _ tokens: MLXArray,
        memory: MLXArray,
        cache: [DecoderLayerCache]? = nil,
        returnCrossQK: Bool = false
    ) -> Output {
        var x = embed_tokens(tokens)

        let layerCache = cache ?? [DecoderLayerCache](repeating: DecoderLayerCache(), count: layers.count)
        var newCache = [DecoderLayerCache](repeating: DecoderLayerCache(), count: layers.count)
        var crossQKAll: [[MLXArray]]? = returnCrossQK ? [] : nil

        for i in 0 ..< layers.count {
            let result = layers[i](
                x, memory: memory,
                selfCache: layerCache[i].selfCache,
                crossCache: layerCache[i].crossCache,
                returnCrossWeights: returnCrossQK
            )
            x = result.hidden
            newCache[i] = DecoderLayerCache(
                selfCache: result.selfCache,
                crossCache: result.crossCache
            )
            if returnCrossQK, let qk = result.crossQK {
                crossQKAll?.append([qk])
            }
        }

        return Output(hidden: norm(x), cache: newCache, crossQKAll: crossQKAll)
    }
}

// MARK: - Decoder Layer Cache

public struct DecoderLayerCache: @unchecked Sendable {
    var selfCache: (MLXArray, MLXArray)?
    var crossCache: (MLXArray, MLXArray)?

    init(selfCache: (MLXArray, MLXArray)? = nil, crossCache: (MLXArray, MLXArray)? = nil) {
        self.selfCache = selfCache
        self.crossCache = crossCache
    }
}
