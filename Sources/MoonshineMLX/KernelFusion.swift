import Foundation
import MLX
import MLXFast

// MARK: - Fused Metal Kernels for Moonshine

/// Custom Metal kernels for fusing hot-path operations in the audio frontend.
final class MoonshineKernelFusion: @unchecked Sendable {
    static let shared = MoonshineKernelFusion()

    // Fused CMVN + Asinh + SiLU kernel (float32)
    private let fusedFrontendF32: MLXFast.MLXFastKernel

    // Fused CMVN + Asinh + SiLU kernel (float16)
    private let fusedFrontendF16: MLXFast.MLXFastKernel

    private init() {
        // Fused per-frame CMVN + asinh compression kernel
        // Input: frames [B*T, frame_len], log_k scalar
        // Output: normalized and compressed frames
        fusedFrontendF32 = MLXFast.metalKernel(
            name: "moonshine_fused_cmvn_asinh_f32",
            inputNames: ["frames", "log_k"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint frame_len = frames_shape[1];
                uint total_frames = frames_shape[0];
                if (idx >= total_frames) return;

                uint base = idx * frame_len;
                float k = exp(log_k[0]);

                // Compute mean
                float sum = 0.0f;
                for (uint i = 0; i < frame_len; i++) {
                    sum += frames[base + i];
                }
                float mean = sum / (float)frame_len;

                // Compute variance
                float var_sum = 0.0f;
                for (uint i = 0; i < frame_len; i++) {
                    float d = frames[base + i] - mean;
                    var_sum += d * d;
                }
                float rms = sqrt(var_sum / (float)frame_len + 1e-6f);

                // CMVN + asinh(k * x)
                for (uint i = 0; i < frame_len; i++) {
                    float normed = (frames[base + i] - mean) / rms;
                    out[base + i] = asinh(k * normed);
                }
            """,
            ensureRowContiguous: true
        )

        fusedFrontendF16 = MLXFast.metalKernel(
            name: "moonshine_fused_cmvn_asinh_f16",
            inputNames: ["frames", "log_k"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint frame_len = frames_shape[1];
                uint total_frames = frames_shape[0];
                if (idx >= total_frames) return;

                uint base = idx * frame_len;
                float k = exp((float)log_k[0]);

                // Compute mean
                float sum = 0.0f;
                for (uint i = 0; i < frame_len; i++) {
                    sum += (float)frames[base + i];
                }
                float mean = sum / (float)frame_len;

                // Compute variance
                float var_sum = 0.0f;
                for (uint i = 0; i < frame_len; i++) {
                    float d = (float)frames[base + i] - mean;
                    var_sum += d * d;
                }
                float rms = sqrt(var_sum / (float)frame_len + 1e-6f);

                // CMVN + asinh(k * x)
                for (uint i = 0; i < frame_len; i++) {
                    float normed = ((float)frames[base + i] - mean) / rms;
                    out[base + i] = asinh(k * normed);
                }
            """,
            ensureRowContiguous: true
        )
    }

    /// Fused CMVN + asinh compression.
    /// Input: frames [B*T, frame_len], log_k: scalar MLXArray.
    /// Returns: [B*T, frame_len] with CMVN normalization and asinh compression applied.
    func fusedCMVNAsinh(frames: MLXArray, logK: MLXArray, threadGroupSize: Int = 256) -> MLXArray? {
        guard frames.ndim == 2 else { return nil }
        guard frames.dtype == .float32 || frames.dtype == .float16 else { return nil }

        let totalFrames = frames.dim(0)
        let tg = max(32, min(threadGroupSize, 1024))
        let kernel = frames.dtype == .float16 ? fusedFrontendF16 : fusedFrontendF32
        let logKCast = logK.asType(frames.dtype)

        return kernel(
            [frames, logKCast],
            grid: (totalFrames, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [frames.shape],
            outputDTypes: [frames.dtype]
        )[0]
    }
}
