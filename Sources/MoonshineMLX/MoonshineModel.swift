import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Transcription Output

/// Result of a transcription operation.
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

/// Moonshine V2 Streaming speech-to-text model.
///
/// Native MLX/Metal implementation targeting `UsefulSensors/moonshine-streaming-{tiny,small,medium}`.
public final class MoonshineModel: Module, @unchecked Sendable {
    public let config: MoonshineModelConfig

    let encoder: MoonshineEncoder
    let decoder: MoonshineDecoder
    let proj_out: Linear

    /// Tokenizer (set after loading).
    public var tokenizer: MoonshineTokenizer?

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
            // Use embedding weights as linear projection
            return hidden.matmul(decoder.embed_tokens.weight.transposed())
        }
        return proj_out(hidden)
    }

    // MARK: - Offline (Batch) Generation

    /// Transcribe audio in a single pass.
    ///
    /// - Parameters:
    ///   - audio: Raw audio samples at 16 kHz (1-D MLXArray), or path string.
    ///   - maxTokens: Maximum tokens to generate.
    ///   - temperature: Sampling temperature (0 = greedy argmax).
    /// - Returns: Transcription result.
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

        // Encode
        let features = encoder.embedder(input)
        let encoded = encoder(features)
        let memory = decoder.prepareMemory(encoded)
        eval(memory)

        // Limit tokens by audio duration
        let dur = Double(input.dim(0)) / Double(sampleRate)
        let maxTok = min(maxTokens, Int(ceil(dur * config.maxTokensPerSecond)))

        // Decode
        var tokens: [Int] = [config.decoderStartTokenId]
        var cache: [DecoderLayerCache]? = nil

        // First token (sync)
        var tok = MLXArray(Int32(tokens.last!)).reshaped(1, 1)
        var result = decoder(tok, memory: memory, cache: cache)
        cache = result.cache
        var logits = getLogits(result.hidden[0..., (-1)..., 0...])
        var nextTokArr = temperature > 0
            ? MLXRandom.categorical(logits / MLXArray(temperature))
            : argMax(logits, axis: -1)
        eval(nextTokArr)

        for _ in 0 ..< (maxTok - 1) {
            let nt = nextTokArr.item(Int.self)
            if nt == config.eosTokenId { break }
            tokens.append(nt)

            tok = MLXArray(Int32(nt)).reshaped(1, 1)
            result = decoder(tok, memory: memory, cache: cache)
            cache = result.cache
            logits = getLogits(result.hidden[0..., (-1)..., 0...])
            nextTokArr = temperature > 0
                ? MLXRandom.categorical(logits / MLXArray(temperature))
                : argMax(logits, axis: -1)
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

    /// Transcribe audio with word-level timestamps via cross-attention DTW.
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

        // Encode
        let features = encoder.embedder(input)
        let encoded = encoder(features)
        let memory = decoder.prepareMemory(encoded)
        eval(memory)

        let numFrames = memory.dim(1)
        let dur = Double(input.dim(0)) / Double(sampleRate)
        let maxTok = min(maxTokens, Int(ceil(dur * config.maxTokensPerSecond)))

        // Decode with cross-attention weight extraction
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

        // Extract word timestamps
        let wordTimings = findAlignment(
            crossQKPerStep: crossQKPerStep,
            tokens: gen,
            tokenToString: { [weak self] id in self?.tokenizer?.tokenToString(id) },
            decodeTokens: { [weak self] ids in self?.tokenizer?.decode(ids) ?? "" },
            numFrames: numFrames,
            timeOffset: timeOffset,
            tokenProbs: tokenProbs
        )

        let words = wordTimings.map { wt in
            WordTiming(word: wt.word, tokens: wt.tokens, start: wt.start, end: wt.end, probability: wt.probability)
        }

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
        // Fallback: ASCII decode
        return tokens.map { t in
            t < 128 ? String(UnicodeScalar(t)!) : "<\(t)>"
        }.joined()
    }
}

// MARK: - Convenience Loading

extension MoonshineModel {
    /// Load a Moonshine model from a HuggingFace repo ID or local path.
    public static func load(
        from source: String = MoonshineModelLoader.defaultModelRepo,
        hfToken: String? = nil
    ) throws -> MoonshineModel {
        try MoonshineModelLoader.load(from: source, hfToken: hfToken)
    }
}
