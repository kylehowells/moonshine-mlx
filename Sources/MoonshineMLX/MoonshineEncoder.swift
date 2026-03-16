import Foundation
import MLX
import MLXNN

// MARK: - Audio Frontend Components

/// Per-frame cepstral mean and variance normalization (no learnable params).
final class FrameCMVN: Module, @unchecked Sendable {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, frame_len]
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let rms = MLX.sqrt(MLX.mean(centered * centered, axis: -1, keepDims: true) + eps)
        return centered / rms
    }
}

/// Asinh compression with learnable scale: asinh(exp(log_k) * x).
final class AsinhCompression: Module, @unchecked Sendable {
    var log_k: MLXArray

    override init() {
        self.log_k = MLXArray(Float(0.0))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.asinh(MLX.exp(log_k) * x)
    }
}

// MARK: - Embedder (Audio Frontend)

/// Converts raw 16 kHz audio into feature frames.
///
/// Architecture:
///   1. Chunk audio into frame_len-sample frames (80 samples = 5ms)
///   2. cmvn: per-frame mean/variance normalization
///   3. comp: asinh(exp(log_k) * x)
///   4. silu(linear(x)): Linear(frame_len, enc_dim, bias=False) + SiLU
///   5. silu(conv1(x)): Conv1d(enc_dim, 2*enc_dim, k=5, s=2) + SiLU
///   6. conv2(x): Conv1d(2*enc_dim, enc_dim, k=5, s=2)
///
/// Output rate: 50 Hz (one feature every 20 ms).
final class Embedder: Module, @unchecked Sendable {
    let cmvn: FrameCMVN
    let comp: AsinhCompression
    let linear: Linear
    let conv1: Conv1d
    let conv2: Conv1d
    let frameLen: Int
    let dim: Int

    init(config: MoonshineModelConfig) {
        let ec = config.encoder
        let dim = ec.hiddenSize
        self.frameLen = ec.frameLen
        self.dim = dim

        self.cmvn = FrameCMVN()
        self.comp = AsinhCompression()
        self.linear = Linear(ec.frameLen, dim, bias: false)
        self.conv1 = Conv1d(inputChannels: dim, outputChannels: 2 * dim, kernelSize: 5, stride: 2, bias: true)
        self.conv2 = Conv1d(inputChannels: 2 * dim, outputChannels: dim, kernelSize: 5, stride: 2, bias: true)
    }

    /// Batch forward. audio: [B, N] or [N].
    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        var input = audio
        if input.ndim == 1 {
            input = input.expandedDimensions(axis: 0)
        }
        let B = input.dim(0)
        let N = input.dim(1)
        let nFrames = N / frameLen
        let trimmed = input[0..., ..<(nFrames * frameLen)]
        let frames = trimmed.reshaped(B, nFrames, frameLen)

        var x = cmvn(frames)
        x = comp(x)
        x = silu(linear(x))

        // Causal padding for conv1 (kernel=5, stride=2)
        x = padded(x, widths: [.init(0), .init((4, 0)), .init(0)])
        x = silu(conv1(x))

        // Causal padding for conv2
        x = padded(x, widths: [.init(0), .init((4, 0)), .init(0)])
        x = conv2(x)

        return x
    }

    /// Streaming forward for a single chunk of audio. Returns (features, updatedState).
    func processChunk(_ audioChunk: MLXArray, state: StreamingState) -> MLXArray {
        var chunk = audioChunk

        // Prepend leftover samples
        if state.sampleLen > 0 {
            let buf = state.sampleBuffer![0 ..< state.sampleLen]
            chunk = concatenated([buf, chunk], axis: 0)
        }

        let nSamples = chunk.dim(0)
        let nFrames = nSamples / frameLen
        let remainder = nSamples - nFrames * frameLen

        if remainder > 0 {
            state.sampleBuffer = chunk[(nSamples - remainder)...]
            state.sampleLen = remainder
        } else {
            state.sampleLen = 0
        }

        if nFrames == 0 {
            return MLXArray.zeros([1, 0, dim])
        }

        let trimmed = chunk[..<(nFrames * frameLen)]
        let frames = trimmed.reshaped(1, nFrames, frameLen)
        var x = cmvn(frames)
        x = comp(x)
        x = silu(linear(x))

        // Conv1 with causal buffer (4 frames history, stride=2)
        let xIn1: MLXArray
        if let buf = state.conv1Buffer {
            xIn1 = concatenated([buf, x], axis: 1)
        } else {
            xIn1 = padded(x, widths: [.init(0), .init((4, 0)), .init(0)])
        }
        state.conv1Buffer = xIn1[0..., (xIn1.dim(1) - 4)..., 0...]
        x = silu(conv1(xIn1))

        // Conv2 with causal buffer
        let xIn2: MLXArray
        if let buf = state.conv2Buffer {
            xIn2 = concatenated([buf, x], axis: 1)
        } else {
            xIn2 = padded(x, widths: [.init(0), .init((4, 0)), .init(0)])
        }
        state.conv2Buffer = xIn2[0..., (xIn2.dim(1) - 4)..., 0...]
        x = conv2(xIn2)

        return x
    }
}

// MARK: - Encoder

final class MoonshineEncoder: Module, @unchecked Sendable {
    let embedder: Embedder
    let layers: [EncoderLayer]
    let final_norm: UnitOffsetLayerNorm

    init(config: MoonshineModelConfig) {
        let ec = config.encoder
        self.embedder = Embedder(config: config)
        self.layers = (0 ..< ec.numHiddenLayers).map { i in
            EncoderLayer(config: config, layerIdx: i)
        }
        self.final_norm = UnitOffsetLayerNorm(ec.hiddenSize)
    }

    /// Encode pre-extracted features.
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = features
        for layer in layers {
            x = layer(x)
        }
        return final_norm(x)
    }
}
