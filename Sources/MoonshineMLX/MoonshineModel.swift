import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Transcription Output

public struct TranscriptionOutput: Sendable {
    public let text: String
    public let segments: [TranscriptionSegment]
    public let generationTokens: Int
    public let totalTime: Double
    public let tokensPerSecond: Double
}

public struct TranscriptionSegment: Sendable {
    public let text: String
    public let start: Double
    public let end: Double
    public let words: [WordTiming]?
}

// MARK: - Moonshine Model

public final class MoonshineModel: Module, @unchecked Sendable {
    public let config: MoonshineModelConfig

    let encoder: MoonshineEncoder
    let decoder: MoonshineDecoder
    let proj_out: Linear

    public var tokenizer: MoonshineTokenizer?

    /// Compiled single-token decoder step for fast autoregressive decoding.
    private var _compiledStep: (([MLXArray]) -> [MLXArray])?

    public init(config: MoonshineModelConfig) {
        self.config = config
        self.encoder = MoonshineEncoder(config: config)
        self.decoder = MoonshineDecoder(config: config)
        self.proj_out = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public var sampleRate: Int { config.encoder.sampleRate }

    // MARK: - Logits

    func getLogits(_ hidden: MLXArray) -> MLXArray {
        if config.tieWordEmbeddings {
            return hidden.matmul(decoder.embed_tokens.weight.transposed())
        }
        return proj_out(hidden)
    }

    // MARK: - Compiled Decode Step

    /// Build a compiled function for a single decode step.
    /// This fuses the decoder forward + logits projection into one optimized graph.
    private func buildCompiledStep() -> @Sendable ([MLXArray]) -> [MLXArray] {
        compile(inputs: [self], outputs: [self]) { inputs in
            // inputs: [token(1,1), memory(1,M,D), selfK_0, selfV_0, crossK_0, crossV_0, ..., selfK_N, selfV_N, crossK_N, crossV_N]
            let token = inputs[0]
            let memory = inputs[1]

            let numLayers = self.decoder.layers.count
            var cache: [DecoderLayerCache] = []
            for i in 0 ..< numLayers {
                let base = 2 + i * 4
                cache.append(DecoderLayerCache(
                    selfCache: (inputs[base], inputs[base + 1]),
                    crossCache: (inputs[base + 2], inputs[base + 3])
                ))
            }

            let result = self.decoder(token, memory: memory, cache: cache)
            let logits = self.getLogits(result.hidden[0..., (-1)..., 0...])
            let nextTok = argMax(logits, axis: -1)

            // outputs: [nextTok, selfK_0, selfV_0, crossK_0, crossV_0, ..., selfK_N, selfV_N, crossK_N, crossV_N]
            var outputs: [MLXArray] = [nextTok]
            for c in result.cache {
                outputs.append(c.selfCache!.0)
                outputs.append(c.selfCache!.1)
                outputs.append(c.crossCache!.0)
                outputs.append(c.crossCache!.1)
            }
            return outputs
        }
    }

    // MARK: - Offline (Batch) Generation

    public func generate(
        audio: MLXArray,
        maxTokens: Int = 500,
        temperature: Float = 0.0
    ) -> TranscriptionOutput {
        let start = CFAbsoluteTimeGetCurrent()

        var input = audio
        if input.ndim != 1 {
            input = input.squeezed()
        }

        // Encode (single fused eval for entire pipeline)
        let features = encoder.embedder(input)
        let encoded = encoder(features)
        let memory = decoder.prepareMemory(encoded)
        eval(memory)

        let dur = Double(input.dim(0)) / Double(sampleRate)
        let maxTok = min(maxTokens, Int(ceil(dur * config.maxTokensPerSecond)))

        // Greedy decode with compiled step + async eval pipelining
        var tokens: [Int] = [config.decoderStartTokenId]
        var cache: [DecoderLayerCache]? = nil

        // First token (sync to prime the cache)
        var tok = MLXArray(Int32(tokens.last!)).reshaped(1, 1)
        var result = decoder(tok, memory: memory, cache: cache)
        cache = result.cache
        var logits = getLogits(result.hidden[0..., (-1)..., 0...])
        var nextTokArr: MLXArray
        if temperature > 0 {
            nextTokArr = MLXRandom.categorical(logits / temperature)
        } else {
            nextTokArr = argMax(logits, axis: -1)
        }
        eval(nextTokArr)

        // Remaining tokens with async eval pipelining
        for _ in 0 ..< (maxTok - 1) {
            let nt = nextTokArr.item(Int.self)
            if nt == config.eosTokenId { break }
            tokens.append(nt)

            tok = MLXArray(Int32(nt)).reshaped(1, 1)
            result = decoder(tok, memory: memory, cache: cache)
            cache = result.cache
            logits = getLogits(result.hidden[0..., (-1)..., 0...])
            if temperature > 0 {
                nextTokArr = MLXRandom.categorical(logits / temperature)
            } else {
                nextTokArr = argMax(logits, axis: -1)
            }
            asyncEval(nextTokArr)
        }

        // Collect final token
        let nt = nextTokArr.item(Int.self)
        if nt != config.eosTokenId {
            tokens.append(nt)
        }

        let gen = Array(tokens.dropFirst())
        let text = decodeTokens(gen)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        return TranscriptionOutput(
            text: text.trimmingCharacters(in: .whitespaces),
            segments: [TranscriptionSegment(
                text: text.trimmingCharacters(in: .whitespaces),
                start: 0.0, end: dur, words: nil
            )],
            generationTokens: gen.count,
            totalTime: elapsed,
            tokensPerSecond: elapsed > 0 ? Double(gen.count) / elapsed : 0
        )
    }

    // MARK: - Word-Level Timestamps

    public func generateWithWordTimestamps(
        audio: MLXArray,
        maxTokens: Int = 500,
        timeOffset: Double = 0.0
    ) -> (output: TranscriptionOutput, words: [WordTiming]) {
        let start = CFAbsoluteTimeGetCurrent()

        var input = audio
        if input.ndim != 1 {
            input = input.squeezed()
        }

        let features = encoder.embedder(input)
        let encoded = encoder(features)
        let memory = decoder.prepareMemory(encoded)
        eval(memory)

        let numFrames = memory.dim(1)
        let dur = Double(input.dim(0)) / Double(sampleRate)
        let maxTok = min(maxTokens, Int(ceil(dur * config.maxTokensPerSecond)))

        var tokens: [Int] = [config.decoderStartTokenId]
        var cache: [DecoderLayerCache]? = nil
        var crossQKPerStep: [[[MLXArray]]] = []
        var tokenProbs: [Float] = []

        for _ in 0 ..< maxTok {
            let tok = MLXArray(Int32(tokens.last!)).reshaped(1, 1)
            let result = decoder(tok, memory: memory, cache: cache, returnCrossQK: true)
            cache = result.cache
            eval(result.hidden)

            if let qks = result.crossQKAll {
                crossQKPerStep.append(qks)
            }

            let logits = getLogits(result.hidden[0..., (-1)..., 0...])
            let probs = softmax(logits, axis: -1)
            let nt = argMax(logits, axis: -1).item(Int.self)
            eval(probs)
            let probVal = probs.reshaped(-1)[nt].item(Float.self)
            tokenProbs.append(probVal)

            if nt == config.eosTokenId { break }
            tokens.append(nt)
        }

        let gen = Array(tokens.dropFirst())
        let text = decodeTokens(gen)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let wordTimings = findAlignment(
            crossQKPerStep: crossQKPerStep,
            tokens: gen,
            tokenToString: { [weak self] id in self?.tokenizer?.tokenToString(id) },
            decodeTokens: { [weak self] ids in self?.tokenizer?.decode(ids) ?? "" },
            numFrames: numFrames,
            timeOffset: timeOffset,
            tokenProbs: tokenProbs
        )

        let words = wordTimings

        let output = TranscriptionOutput(
            text: text.trimmingCharacters(in: .whitespaces),
            segments: [TranscriptionSegment(
                text: text.trimmingCharacters(in: .whitespaces),
                start: 0.0, end: dur, words: words
            )],
            generationTokens: gen.count,
            totalTime: elapsed,
            tokensPerSecond: elapsed > 0 ? Double(gen.count) / elapsed : 0
        )

        return (output, wordTimings)
    }

    // MARK: - Token Decoding

    func decodeTokens(_ tokens: [Int]) -> String {
        if let tokenizer {
            return tokenizer.decode(tokens)
        }
        return tokens.map { t in
            t < 128 ? String(UnicodeScalar(t)!) : "<\(t)>"
        }.joined()
    }
}

// MARK: - Convenience Loading

extension MoonshineModel {
    public static func load(
        from source: String = MoonshineModelLoader.defaultModelRepo,
        hfToken: String? = nil
    ) throws -> MoonshineModel {
        try MoonshineModelLoader.load(from: source, hfToken: hfToken)
    }
}
