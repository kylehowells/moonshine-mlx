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

    public let encoder: MoonshineEncoder
    public let decoder: MoonshineDecoder
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

    /// Pre-compute cached values after weights are loaded.
    func prepareForInference() {
        // Pre-compute UnitOffsetLayerNorm offset weights
        for layer in encoder.layers {
            layer.input_layernorm.prepareForInference()
            layer.post_attention_layernorm.prepareForInference()
        }
        encoder.final_norm.prepareForInference()
    }

    // MARK: - Logits

    /// Project decoder hidden state to vocabulary logits.
    /// Input: [B, T, D] — extracts last timestep, returns [B, vocab].
    public func getLogits(_ hidden: MLXArray) -> MLXArray {
        // Select last timestep: [B, T, D] -> [B, D] (matching Python's hidden[:, -1, :])
        let h: MLXArray
        if hidden.ndim == 3 {
            let B = hidden.dim(0)
            let D = hidden.dim(2)
            h = hidden[0..., hidden.dim(1) - 1, 0...].reshaped(B, D)
        } else {
            h = hidden
        }
        if config.tieWordEmbeddings {
            return h.matmul(decoder.embed_tokens.weight.transposed())
        }
        return proj_out(h)
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
            let logits = self.getLogits(result.hidden)
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

    /// Transcribe audio of any length.
    ///
    /// Audio is processed in `chunkDuration`-second segments through the
    /// batch encoder (full sliding-window attention per chunk). Each chunk
    /// is encoded and decoded independently - matching the Python reference.
    public func generate(
        audio: MLXArray,
        maxTokens: Int = Int.max,
        temperature: Float = 0.0,
        chunkDuration: Double = 30.0,
        profile: Bool = false
    ) -> TranscriptionOutput {
        let startTime = CFAbsoluteTimeGetCurrent()

        var input = audio
        if input.ndim != 1 { input = input.squeezed() }

        let totalSamples = input.dim(0)
        let dur = Double(totalSamples) / Double(sampleRate)
        let chunkSamples = Int(chunkDuration * Double(sampleRate))

        var allText: [String] = []
        var totalTokens = 0
        var tEncodeTotal = 0.0
        var tDecodeTotal = 0.0

        for offset in stride(from: 0, to: totalSamples, by: chunkSamples) {
            let end = min(offset + chunkSamples, totalSamples)
            let chunk = input[offset ..< end]

            let (text, nTok, tEnc, tDec) = encodeAndDecodeChunk(
                chunk, maxTokens: maxTokens, temperature: temperature
            )
            if !text.isEmpty { allText.append(text) }
            totalTokens += nTok
            tEncodeTotal += tEnc
            tDecodeTotal += tDec
        }

        let fullText = allText.joined(separator: " ").trimmingCharacters(in: .whitespaces)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime

        if profile {
            FileHandle.standardError.write(Data("""
            PROFILE (Swift MLX):
              Encode:      \(String(format: "%7.1f", tEncodeTotal * 1000))ms
              Decode:      \(String(format: "%7.1f", tDecodeTotal * 1000))ms (\(totalTokens) tokens)
              Total:       \(String(format: "%7.1f", elapsed * 1000))ms
              Tokens: \(totalTokens), RTF: \(String(format: "%.4f", elapsed / dur))x\n
            """.utf8))
        }

        return TranscriptionOutput(
            text: fullText,
            segments: [TranscriptionSegment(text: fullText, start: 0.0, end: dur, words: nil)],
            generationTokens: totalTokens,
            totalTime: elapsed,
            tokensPerSecond: elapsed > 0 ? Double(totalTokens) / elapsed : 0
        )
    }

    /// Encode + decode a single audio chunk. Returns (text, tokenCount, encodeTime, decodeTime).
    private func encodeAndDecodeChunk(
        _ audio: MLXArray, maxTokens: Int, temperature: Float
    ) -> (String, Int, Double, Double) {
        var input = audio
        if input.ndim == 1 { input = input.expandedDimensions(axis: 0) }

        let t0 = CFAbsoluteTimeGetCurrent()
        let features = encoder.embedder(input)
        let encoded = encoder(features)
        let memory = decoder.prepareMemory(encoded)
        eval(memory)
        let tEncode = CFAbsoluteTimeGetCurrent() - t0

        let dur = Double(audio.dim(audio.ndim - 1)) / Double(sampleRate)
        let limit = min(maxTokens, Int(ceil(dur * config.maxTokensPerSecond)))

        let t1 = CFAbsoluteTimeGetCurrent()
        var tokens: [Int] = [config.decoderStartTokenId]
        var cache: [DecoderLayerCache]? = nil

        var tok = MLXArray(Int32(tokens.last!)).reshaped(1, 1)
        var result = decoder(tok, memory: memory, cache: cache)
        cache = result.cache
        var logits = getLogits(result.hidden)
        var nextTokArr = temperature > 0
            ? MLXRandom.categorical(logits / temperature)
            : argMax(logits, axis: -1)
        eval(nextTokArr)

        for _ in 0 ..< (limit - 1) {
            let nt = nextTokArr.item(Int.self)
            if nt == config.eosTokenId { break }
            tokens.append(nt)

            tok = MLXArray(Int32(nt)).reshaped(1, 1)
            result = decoder(tok, memory: memory, cache: cache)
            cache = result.cache
            logits = getLogits(result.hidden)
            nextTokArr = temperature > 0
                ? MLXRandom.categorical(logits / temperature)
                : argMax(logits, axis: -1)
            asyncEval(nextTokArr)
        }

        let nt = nextTokArr.item(Int.self)
        if nt != config.eosTokenId { tokens.append(nt) }
        let tDecode = CFAbsoluteTimeGetCurrent() - t1

        let gen = Array(tokens.dropFirst())
        return (decodeTokens(gen), gen.count, tEncode, tDecode)
    }

    // MARK: - Word-Level Timestamps

    public func generateWithWordTimestamps(
        audio: MLXArray,
        maxTokens: Int = Int.max,
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

            let logits = getLogits(result.hidden)
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

// MARK: - Diagnostic Trace

public struct TraceStep: Sendable {
    public let step: Int
    public let token: Int
    public let logitMax: Float
    public let logitMin: Float
    public let logitMean: Float
}

public struct TraceResult: Sendable {
    public let tokenCount: Int
    public let hitEOS: Bool
    public let memoryShape: [Int]
    public let memoryFirst5: [Float]
    public let memoryLast5: [Float]
    public let steps: [TraceStep]
}

extension MoonshineModel {
    /// Diagnostic: encode + decode step-by-step, dumping per-token stats.
    public func traceGenerate(audio: MLXArray, maxSteps: Int = 200) -> TraceResult {
        var input = audio
        if input.ndim != 1 { input = input.squeezed() }

        let features = encoder.embedder(input)
        let encoded = encoder(features)
        let memory = decoder.prepareMemory(encoded)
        eval(memory)

        let m0 = memory[0, 0, ..<5]
        let mLast = memory[0, memory.dim(1) - 1, ..<5]
        eval(m0, mLast)

        var tokens: [Int] = [config.decoderStartTokenId]
        var cache: [DecoderLayerCache]? = nil
        var steps: [TraceStep] = []
        var hitEOS = false

        for step in 0 ..< maxSteps {
            let tok = MLXArray(Int32(tokens.last!)).reshaped(1, 1)
            let result = decoder(tok, memory: memory, cache: cache)
            cache = result.cache
            let logits = getLogits(result.hidden)
            eval(logits)

            let flat = logits.reshaped(-1)
            let nt = argMax(flat, axis: 0).item(Int.self)
            let topVal = flat[nt].item(Float.self)
            let logMin = flat.min().item(Float.self)
            let logMax = flat.max().item(Float.self)
            let logMean = flat.mean().item(Float.self)

            steps.append(TraceStep(
                step: step, token: nt,
                logitMax: logMax, logitMin: logMin, logitMean: logMean
            ))

            if nt == config.eosTokenId {
                hitEOS = true
                break
            }
            tokens.append(nt)
        }

        return TraceResult(
            tokenCount: tokens.count - 1,
            hitEOS: hitEOS,
            memoryShape: memory.shape.map { $0 },
            memoryFirst5: m0.asArray(Float.self),
            memoryLast5: mLast.asArray(Float.self),
            steps: steps
        )
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
