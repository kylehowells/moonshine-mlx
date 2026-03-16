import Foundation
import MLX
import MLXRandom

// MARK: - Streaming State

/// Mutable state for streaming transcription.
public final class StreamingState: @unchecked Sendable {
    // Frontend (persists across segments - causal conv state)
    var sampleBuffer: MLXArray?
    var sampleLen: Int = 0
    var conv1Buffer: MLXArray?
    var conv2Buffer: MLXArray?

    // Accumulated encoder-input features [1, T, enc_dim]
    var accumulatedFeatures: MLXArray?
    var accumulatedFeatureCount: Int = 0

    // Encoder progress
    var encoderFramesEmitted: Int = 0

    // Positional offset for pos_emb (continuous across segments)
    var posOffset: Int = 0

    // Memory [1, M, dec_dim]
    var memory: MLXArray?
    var memoryLen: Int = 0

    // Decoder cache (reset each transcribe)
    var decoderCache: [DecoderLayerCache]?

    var active: Bool = false

    init(sampleBuffer: MLXArray? = nil) {
        self.sampleBuffer = sampleBuffer
    }
}

// MARK: - Streaming API Extension

extension MoonshineModel {

    /// Create a new streaming state.
    public func createStream() -> StreamingState {
        StreamingState(sampleBuffer: MLXArray.zeros([config.encoder.frameLen - 1]))
    }

    /// Start (or restart) a stream, resetting accumulated audio and decoder state.
    /// Frontend convolution buffers are preserved for audio continuity.
    public func startStream(_ state: StreamingState) {
        state.active = true
        state.accumulatedFeatures = nil
        state.accumulatedFeatureCount = 0
        state.encoderFramesEmitted = 0
        state.posOffset = 0
        state.memory = nil
        state.memoryLen = 0
        state.decoderCache = nil
    }

    /// Stop the stream.
    public func stopStream(_ state: StreamingState) {
        state.active = false
    }

    /// Feed audio samples into the streaming pipeline.
    /// - Parameter chunk: 1-D float audio at 16 kHz.
    public func addAudio(_ state: StreamingState, chunk: MLXArray) {
        var input = chunk
        if input.ndim != 1 {
            input = input.squeezed()
        }
        let features = encoder.embedder.processChunk(input, state: state)
        eval(features)

        if features.dim(1) > 0 {
            if state.accumulatedFeatures == nil {
                state.accumulatedFeatures = features
            } else {
                state.accumulatedFeatures = concatenated(
                    [state.accumulatedFeatures!, features], axis: 1
                )
            }
            state.accumulatedFeatureCount = state.accumulatedFeatures!.dim(1)
        }
    }

    /// Transcribe currently accumulated audio.
    ///
    /// - Parameters:
    ///   - isFinal: If true, flush all remaining frames (including lookahead)
    ///     and automatically reset for the next segment. Frontend buffers are
    ///     preserved so audio continuity is maintained.
    ///   - maxTokens: Maximum tokens to generate.
    ///   - temperature: Sampling temperature (0 = greedy).
    /// - Returns: Transcribed text for the current segment.
    public func transcribe(
        _ state: StreamingState,
        isFinal: Bool = false,
        maxTokens: Int = Int.max,
        temperature: Float = 0.0
    ) -> String {
        let ec = config.encoder

        guard let features = state.accumulatedFeatures, state.accumulatedFeatureCount > 0 else {
            if isFinal { resetSegment(state) }
            return ""
        }

        let total = state.accumulatedFeatureCount

        // Determine stable frames (hold back lookahead unless final)
        var lookaheadFrames = 0
        if let windows = ec.slidingWindows {
            lookaheadFrames = windows.map { $0[1] }.max() ?? 0
        }
        let stable = isFinal ? total : max(0, total - lookaheadFrames)
        let newFrames = stable - state.encoderFramesEmitted

        if newFrames <= 0 {
            let result: String
            if state.memoryLen > 0 {
                result = decodeMemory(state, maxTokens: maxTokens, temperature: temperature)
            } else {
                result = ""
            }
            if isFinal { resetSegment(state) }
            return result
        }

        // Encoder: sliding window with left context
        let leftCtx = 16 * ec.numHiddenLayers
        let winStart = max(0, state.encoderFramesEmitted - leftCtx)
        let window = features[0..., winStart ..< total, 0...]

        let encoded = encoder(window)
        let offset = state.encoderFramesEmitted - winStart
        let newEncoded = encoded[0..., offset ..< (offset + newFrames), 0...]
        let newMemory = decoder.prepareMemory(newEncoded, posOffset: state.posOffset)
        eval(newMemory)

        state.posOffset += newFrames
        state.encoderFramesEmitted = stable

        if state.memory == nil {
            state.memory = newMemory
        } else {
            state.memory = concatenated([state.memory!, newMemory], axis: 1)
        }
        state.memoryLen = state.memory!.dim(1)

        let text = decodeMemory(state, maxTokens: maxTokens, temperature: temperature)

        if isFinal {
            resetSegment(state)
        }

        return text
    }

    /// Reset decoder and encoder state for the next segment,
    /// keeping frontend convolution buffers for audio continuity.
    private func resetSegment(_ state: StreamingState) {
        state.accumulatedFeatures = nil
        state.accumulatedFeatureCount = 0
        state.encoderFramesEmitted = 0
        state.posOffset = 0
        state.memory = nil
        state.memoryLen = 0
        state.decoderCache = nil
        // Note: sampleBuffer, conv1Buffer, conv2Buffer are preserved
        // so the next segment picks up seamlessly from the audio stream.
    }

    /// Auto-regressive decode from accumulated memory.
    func decodeMemory(_ state: StreamingState, maxTokens: Int, temperature: Float) -> String {
        guard let memory = state.memory, state.memoryLen > 0 else { return "" }

        let dur = Double(state.memoryLen) * 0.020
        let limit = Int(ceil(dur * config.maxTokensPerSecond))
        let maxTok = min(maxTokens, limit)

        var tokens: [Int] = [config.decoderStartTokenId]
        var cache: [DecoderLayerCache]? = nil

        // First token (sync)
        var tok = MLXArray(Int32(tokens.last!)).reshaped(1, 1)
        var result = decoder(tok, memory: memory, cache: cache)
        cache = result.cache
        var logits = getLogits(result.hidden)
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
            logits = getLogits(result.hidden)
            nextTokArr = temperature > 0
                ? MLXRandom.categorical(logits / MLXArray(temperature))
                : argMax(logits, axis: -1)
            asyncEval(nextTokArr)
        }

        let nt = nextTokArr.item(Int.self)
        if nt != config.eosTokenId {
            tokens.append(nt)
        }

        return decodeTokens(Array(tokens.dropFirst()))
    }
}
