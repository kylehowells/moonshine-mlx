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

    // Positional offset for pos_emb (continuous across the full stream)
    var posOffset: Int = 0

    // Memory [1, M, dec_dim] for current segment
    var memory: MLXArray?
    var memoryLen: Int = 0

    // Decoder cache (reset each transcribe)
    var decoderCache: [DecoderLayerCache]?

    // Segments that have been permanently emitted via EOS or cap
    var emittedSegments: [String] = []

    // Whether last decode ended with EOS (natural sentence boundary)
    var lastDecodeHitEOS: Bool = false

    var active: Bool = false

    init(sampleBuffer: MLXArray? = nil) {
        self.sampleBuffer = sampleBuffer
    }
}

// MARK: - Streaming Configuration

public struct StreamingConfig: Sendable {
    /// Maximum memory frames before forcing a segment break.
    /// 1500 frames = 30s of audio at 50Hz encoder output.
    public var maxMemoryFrames: Int

    /// Minimum memory frames before accepting EOS as a segment boundary.
    /// 250 frames = 5s. Prevents premature segmentation on tiny audio fragments.
    public var minSegmentFrames: Int

    public init(maxMemoryFrames: Int = 100000, minSegmentFrames: Int = 250) {
        self.maxMemoryFrames = maxMemoryFrames
        self.minSegmentFrames = minSegmentFrames
    }

    public static let `default` = StreamingConfig()
}

// MARK: - Streaming API

extension MoonshineModel {

    /// Create a new streaming state.
    public func createStream() -> StreamingState {
        StreamingState(sampleBuffer: MLXArray.zeros([config.encoder.frameLen - 1]))
    }

    /// Start (or restart) a stream. Frontend convolution buffers are preserved.
    public func startStream(_ state: StreamingState) {
        state.active = true
        resetSegment(state)
        state.emittedSegments = []
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
    /// The decoder re-decodes from BOS each call (with cached cross-attention KV).
    /// When the model produces EOS (natural sentence boundary), the text is
    /// permanently emitted and the segment resets automatically.
    /// A memory cap (default 30s) forces a segment break if EOS doesn't come.
    ///
    /// - Parameters:
    ///   - isFinal: If true, flush all frames and emit remaining text.
    ///   - streamingConfig: Controls memory cap.
    ///   - maxTokens: Maximum tokens per decode pass.
    ///   - temperature: Sampling temperature (0 = greedy).
    /// - Returns: Current transcription text.
    ///   For partial calls: full text of current segment (for live display).
    ///   After EOS or isFinal: the newly emitted text.
    public func transcribe(
        _ state: StreamingState,
        isFinal: Bool = false,
        streamingConfig: StreamingConfig = .default,
        maxTokens: Int = Int.max,
        temperature: Float = 0.0
    ) -> String {
        // If the previous call emitted via EOS, the segment was already reset.
        // Check if we need to encode new frames.
        encodeNewFrames(state, isFinal: isFinal)

        guard state.memoryLen > 0 else {
            if isFinal { resetSegment(state) }
            return ""
        }

        // Decode from BOS against current memory
        let (text, hitEOS) = decodeMemory(
            state, maxTokens: maxTokens, temperature: temperature
        )
        state.lastDecodeHitEOS = hitEOS

        if hitEOS && state.memoryLen >= streamingConfig.minSegmentFrames {
            // Natural sentence boundary with enough context - emit and reset
            state.emittedSegments.append(text)
            resetSegment(state)
            encodeNewFrames(state, isFinal: isFinal)
            return text
        }

        if isFinal {
            // End of stream - emit whatever we have
            state.emittedSegments.append(text)
            resetSegment(state)
            return text
        }

        // Safety cap: force segment break if memory too large
        if state.memoryLen > streamingConfig.maxMemoryFrames {
            state.emittedSegments.append(text)
            resetSegment(state)
            return text
        }

        // Partial result for live display
        return text
    }

    /// Get all permanently emitted text.
    public func getEmittedText(_ state: StreamingState) -> String {
        state.emittedSegments.joined(separator: " ")
    }

    // MARK: - Internal

    /// Encode any new frames from accumulated features into memory.
    private func encodeNewFrames(_ state: StreamingState, isFinal: Bool) {
        let ec = config.encoder

        guard let features = state.accumulatedFeatures, state.accumulatedFeatureCount > 0 else {
            return
        }

        let total = state.accumulatedFeatureCount

        var lookaheadFrames = 0
        if let windows = ec.slidingWindows {
            lookaheadFrames = windows.map { $0[1] }.max() ?? 0
        }
        let stable = isFinal ? total : max(0, total - lookaheadFrames)
        let newFrames = stable - state.encoderFramesEmitted
        guard newFrames > 0 else { return }

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
    }

    /// Reset segment state, keeping frontend buffers and emitted history.
    private func resetSegment(_ state: StreamingState) {
        state.accumulatedFeatures = nil
        state.accumulatedFeatureCount = 0
        state.encoderFramesEmitted = 0
        state.posOffset = 0
        state.memory = nil
        state.memoryLen = 0
        state.decoderCache = nil
        state.lastDecodeHitEOS = false
    }

    /// Decode from BOS against current memory. Returns (text, hitEOS).
    func decodeMemory(
        _ state: StreamingState, maxTokens: Int, temperature: Float
    ) -> (String, Bool) {
        guard let memory = state.memory, state.memoryLen > 0 else { return ("", false) }

        let dur = Double(state.memoryLen) * 0.020
        let limit = Int(ceil(dur * config.maxTokensPerSecond))
        let maxTok = min(maxTokens, limit)

        var tokens: [Int] = [config.decoderStartTokenId]
        var cache: [DecoderLayerCache]? = nil
        var hitEOS = false

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
            if nt == config.eosTokenId { hitEOS = true; break }
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

        if !hitEOS {
            let nt = nextTokArr.item(Int.self)
            if nt == config.eosTokenId {
                hitEOS = true
            } else {
                tokens.append(nt)
            }
        }

        return (decodeTokens(Array(tokens.dropFirst())), hitEOS)
    }
}

// MARK: - Long Audio (Offline)

extension MoonshineModel {
    /// Transcribe audio of any length with bounded memory.
    ///
    /// For offline use: encodes the full audio in one pass (best quality), then
    /// decodes in segments using EOS-driven boundaries with a memory cap.
    ///
    /// Memory usage is bounded by maxMemoryFrames regardless of audio length.
    public func generateLong(
        audio: MLXArray,
        streamingConfig: StreamingConfig = .default
    ) -> TranscriptionOutput {
        let start = CFAbsoluteTimeGetCurrent()

        var input = audio
        if input.ndim != 1 { input = input.squeezed() }

        let dur = Double(input.dim(0)) / Double(sampleRate)

        // Full encode in one pass (gives encoder full context for best quality)
        let features = encoder.embedder(input)
        let encoded = encoder(features)
        let fullMemory = decoder.prepareMemory(encoded)
        eval(fullMemory)

        let totalFrames = fullMemory.dim(1)
        let maxMem = streamingConfig.maxMemoryFrames
        let minSeg = streamingConfig.minSegmentFrames

        var segments: [String] = []
        var memStart = 0

        // Decode in segments: grow memory until EOS or cap, then emit and advance
        while memStart < totalFrames {
            let memEnd = min(memStart + maxMem, totalFrames)
            let memory = fullMemory[0..., memStart ..< memEnd, 0...]
            let memFrames = memEnd - memStart

            let tokLimit = Int(ceil(Double(memFrames) * 0.02 * config.maxTokensPerSecond))

            var tokens: [Int] = [config.decoderStartTokenId]
            var cache: [DecoderLayerCache]? = nil
            var hitEOS = false

            var tok = MLXArray(Int32(tokens.last!)).reshaped(1, 1)
            var result = decoder(tok, memory: memory, cache: cache)
            cache = result.cache
            var logits = getLogits(result.hidden)
            var nextTokArr = argMax(logits, axis: -1)
            eval(nextTokArr)

            for _ in 0 ..< (tokLimit - 1) {
                let nt = nextTokArr.item(Int.self)
                if nt == config.eosTokenId { hitEOS = true; break }
                tokens.append(nt)

                tok = MLXArray(Int32(nt)).reshaped(1, 1)
                result = decoder(tok, memory: memory, cache: cache)
                cache = result.cache
                logits = getLogits(result.hidden)
                nextTokArr = argMax(logits, axis: -1)
                asyncEval(nextTokArr)
            }
            if !hitEOS {
                let nt = nextTokArr.item(Int.self)
                if nt != config.eosTokenId { tokens.append(nt) }
            }

            let text = decodeTokens(Array(tokens.dropFirst()))
            if !text.isEmpty {
                segments.append(text)
            }

            // Advance: if EOS came early, start next segment where EOS occurred
            // Estimate: tokens map ~linearly to frames
            if hitEOS && memFrames > minSeg {
                let framesUsed = max(minSeg, Int(Double(tokens.count) / config.maxTokensPerSecond * 50.0))
                memStart += min(framesUsed, memFrames)
            } else {
                memStart = memEnd
            }
        }

        let fullText = segments.joined(separator: " ")
            .trimmingCharacters(in: .whitespaces)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        return TranscriptionOutput(
            text: fullText,
            segments: [TranscriptionSegment(
                text: fullText, start: 0.0, end: dur, words: nil
            )],
            generationTokens: segments.count,
            totalTime: elapsed,
            tokensPerSecond: elapsed > 0 ? dur / elapsed : 0
        )
    }
}
