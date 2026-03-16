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

    // Rolling window: text that has been finalized and emitted
    var emittedSegments: [String] = []

    var active: Bool = false

    init(sampleBuffer: MLXArray? = nil) {
        self.sampleBuffer = sampleBuffer
    }
}

// MARK: - Streaming Configuration

public struct StreamingConfig: Sendable {
    /// Maximum memory frames before the window rolls forward.
    /// 1500 frames = 30s of audio at 50Hz encoder output.
    public var maxMemoryFrames: Int

    /// Overlap frames kept when rolling (unused in clean-segment mode).
    public var overlapFrames: Int

    public init(maxMemoryFrames: Int = 1500, overlapFrames: Int = 150) {
        self.maxMemoryFrames = maxMemoryFrames
        self.overlapFrames = overlapFrames
    }

    /// Default config: 30s window (matches the sweet spot for Moonshine V2).
    public static let `default` = StreamingConfig()
}

// MARK: - Streaming API

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

    /// Transcribe currently accumulated audio with automatic rolling window.
    ///
    /// When memory exceeds `streamingConfig.maxMemoryFrames`, the window rolls
    /// forward: the oldest portion is decoded, emitted, and dropped. This bounds
    /// memory usage regardless of how long the stream runs.
    ///
    /// - Parameters:
    ///   - isFinal: If true, flush all remaining frames and emit everything.
    ///   - streamingConfig: Controls window size and overlap.
    ///   - maxTokens: Maximum tokens per decode pass.
    ///   - temperature: Sampling temperature (0 = greedy).
    /// - Returns: Newly emitted text (only the new portion, not previously emitted text).
    public func transcribe(
        _ state: StreamingState,
        isFinal: Bool = false,
        streamingConfig: StreamingConfig = .default,
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
            if isFinal, state.memoryLen > 0 {
                let text = decodeMemory(state, maxTokens: maxTokens, temperature: temperature)
                state.emittedSegments.append(text)
                resetSegment(state)
                return text
            }
            if isFinal { resetSegment(state) }
            return state.memoryLen > 0
                ? decodeMemory(state, maxTokens: maxTokens, temperature: temperature)
                : ""
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

        // Rolling window: if memory exceeds max, emit this segment and reset
        // for the next one. This bounds memory regardless of audio length.
        var newlyEmitted = ""
        if !isFinal && state.memoryLen > streamingConfig.maxMemoryFrames {
            // Decode and emit the current segment
            let text = decodeMemory(state, maxTokens: maxTokens, temperature: temperature)
            state.emittedSegments.append(text)
            newlyEmitted = text

            // Reset encoder/decoder state but keep frontend buffers
            // so the next segment picks up seamlessly from the audio stream
            resetSegment(state)
        } else if isFinal {
            let text = decodeMemory(state, maxTokens: maxTokens, temperature: temperature)
            state.emittedSegments.append(text)
            newlyEmitted = text
            resetSegment(state)
        } else {
            // Partial transcription for live display (not permanently emitted)
            return decodeMemory(state, maxTokens: maxTokens, temperature: temperature)
        }

        return newlyEmitted
    }

    /// Get all text emitted so far (concatenation of all finalized segments).
    public func getEmittedText(_ state: StreamingState) -> String {
        state.emittedSegments.joined(separator: " ")
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
    }

    /// Auto-regressive decode from accumulated memory.
    func decodeMemory(_ state: StreamingState, maxTokens: Int, temperature: Float) -> String {
        guard let memory = state.memory, state.memoryLen > 0 else { return "" }

        let dur = Double(state.memoryLen) * 0.020
        let limit = Int(ceil(dur * config.maxTokensPerSecond))
        let maxTok = min(maxTokens, limit)

        var tokens: [Int] = [config.decoderStartTokenId]
        var cache: [DecoderLayerCache]? = nil

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

// MARK: - Long Audio (Offline)

extension MoonshineModel {
    /// Transcribe audio of any length by processing it in chunks through the
    /// streaming pipeline with a rolling memory window.
    ///
    /// This handles hour-long files with bounded memory usage.
    ///
    /// - Parameters:
    ///   - audio: Raw audio samples at 16 kHz.
    ///   - chunkDuration: Seconds of audio to feed per chunk (default 0.5s).
    ///   - streamingConfig: Controls memory window size.
    /// - Returns: Full transcription.
    public func generateLong(
        audio: MLXArray,
        chunkDuration: Double = 0.5,
        streamingConfig: StreamingConfig = .default
    ) -> TranscriptionOutput {
        let start = CFAbsoluteTimeGetCurrent()

        var input = audio
        if input.ndim != 1 { input = input.squeezed() }

        let totalSamples = input.dim(0)
        let dur = Double(totalSamples) / Double(sampleRate)
        let chunkSamples = Int(chunkDuration * Double(sampleRate))

        let state = createStream()
        startStream(state)

        var offset = 0

        while offset < totalSamples {
            let end = min(offset + chunkSamples, totalSamples)
            let chunk = input[offset ..< end]
            let isLast = end >= totalSamples

            addAudio(state, chunk: chunk)

            // transcribe handles the rolling window internally:
            // - when memory overflows, it emits the segment and trims
            // - on isFinal, it emits the remaining text
            // We discard the return value for non-emitting calls (partials).
            _ = transcribe(
                state, isFinal: isLast,
                streamingConfig: streamingConfig
            )

            offset = end
        }

        // All emitted segments are collected in state.emittedSegments
        let fullText = getEmittedText(state)
            .trimmingCharacters(in: .whitespaces)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let approxTokens = Int(dur * config.maxTokensPerSecond)

        return TranscriptionOutput(
            text: fullText,
            segments: [TranscriptionSegment(
                text: fullText, start: 0.0, end: dur, words: nil
            )],
            generationTokens: approxTokens,
            totalTime: elapsed,
            tokensPerSecond: elapsed > 0 ? Double(approxTokens) / elapsed : 0
        )
    }
}
