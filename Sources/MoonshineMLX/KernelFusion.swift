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

    // Fused residual add + LayerNorm (float32)
    // Input: x [N, D], sublayer [N, D], weight [D]
    // Output: residual [N, D] = x + sublayer, normed [N, D] = layernorm(residual)
    private let fusedResidualLayerNormF32: MLXFast.MLXFastKernel

    private init() {
        // Fused residual add + LayerNorm with parallel reduction.
        // One threadgroup per row (position), TG_SIZE threads per threadgroup.
        // Each thread handles D/TG_SIZE elements, then parallel reduction for mean/var.
        fusedResidualLayerNormF32 = MLXFast.metalKernel(
            name: "moonshine_fused_residual_layernorm_f32",
            inputNames: ["x", "sublayer", "weight"],
            outputNames: ["residual", "normed"],
            source: """
                // thread_position_in_grid = (tid_in_threadgroup, row, 0)
                uint tid = thread_position_in_threadgroup.x;
                uint tg_size = threads_per_threadgroup.x;
                uint row = threadgroup_position_in_grid.x;
                uint D = x_shape[x_ndim - 1];
                uint N = 1;
                for (uint i = 0; i < x_ndim - 1; i++) N *= x_shape[i];
                if (row >= N) return;

                uint base = row * D;
                float eps = 1e-5f;

                // Shared memory for reduction
                threadgroup float shared_sum[256];
                threadgroup float shared_var[256];

                // Pass 1: residual add + partial sum
                float local_sum = 0.0f;
                for (uint d = tid; d < D; d += tg_size) {
                    float r = x[base + d] + sublayer[base + d];
                    residual[base + d] = r;
                    local_sum += r;
                }
                shared_sum[tid] = local_sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Parallel reduction for mean
                for (uint s = tg_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_sum[tid] += shared_sum[tid + s];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float mean = shared_sum[0] / (float)D;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Pass 2: partial variance
                float local_var = 0.0f;
                for (uint d = tid; d < D; d += tg_size) {
                    float diff = residual[base + d] - mean;
                    local_var += diff * diff;
                }
                shared_var[tid] = local_var;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Parallel reduction for variance
                for (uint s = tg_size / 2; s > 0; s >>= 1) {
                    if (tid < s) shared_var[tid] += shared_var[tid + s];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                float inv_std = rsqrt(shared_var[0] / (float)D + eps);

                // Pass 3: normalize with weight
                for (uint d = tid; d < D; d += tg_size) {
                    normed[base + d] = (residual[base + d] - mean) * inv_std * weight[d];
                }
            """,
            ensureRowContiguous: true
        )

        // Fused SiLU-gated linear unit: out[i] = silu(x[i + half]) * x[i]
        fusedSwiGLUGateF32 = MLXFast.metalKernel(
            name: "moonshine_swiglu_gate_f32",
            inputNames: ["projected"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint B_T = projected_shape[0] * projected_shape[1];
                uint full_dim = projected_shape[2];
                uint half_dim = full_dim / 2;
                uint total = B_T * half_dim;
                if (idx >= total) return;

                uint bt = idx / half_dim;
                uint d = idx % half_dim;
                uint base = bt * full_dim;

                float x_val = projected[base + d];
                float gate = projected[base + half_dim + d];
                // silu(gate) = gate * sigmoid(gate)
                float sig = 1.0f / (1.0f + exp(-gate));
                out[idx] = x_val * gate * sig;
            """,
            ensureRowContiguous: true
        )

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

    // MARK: - Fused Residual + LayerNorm

    /// Fused residual add + LayerNorm.
    /// Computes: residual = x + sublayer; normed = layernorm(residual, weight)
    /// Returns (residual, normed) — saving one GPU dispatch vs separate ops.
    func fusedResidualLayerNorm(
        x: MLXArray, sublayer: MLXArray, weight: MLXArray
    ) -> (residual: MLXArray, normed: MLXArray)? {
        guard x.dtype == .float32, x.shape == sublayer.shape else { return nil }
        let shape = x.shape
        // Total rows = product of all dims except last
        let N = shape.dropLast().reduce(1, *)
        let D = shape.last!

        // One threadgroup per row, 128 threads per threadgroup for parallel reduction
        let tgSize = min(128, D)
        let results = fusedResidualLayerNormF32(
            [x, sublayer, weight],
            grid: (tgSize * N, 1, 1),
            threadGroup: (tgSize, 1, 1),
            outputShapes: [shape, shape],
            outputDTypes: [x.dtype, x.dtype]
        )
        return (results[0], results[1])
    }

    // MARK: - Fused SiLU-Gated Linear (for SwiGLU FFN)

    private let fusedSwiGLUGateF32: MLXFast.MLXFastKernel

    /// Fused silu(gate) * x operation for the SwiGLU FFN.
    /// Input: projected [B, T, 2*intermediate] from fc1
    /// Output: [B, T, intermediate] = silu(second_half) * first_half
    func fusedSwiGLUGate(projected: MLXArray, intermediateSize: Int, threadGroupSize: Int = 256) -> MLXArray? {
        guard projected.dtype == .float32 else { return nil }
        guard projected.ndim == 3 else { return nil }

        let B = projected.dim(0)
        let T = projected.dim(1)
        let total = B * T * intermediateSize
        let tg = max(32, min(threadGroupSize, 1024))

        return fusedSwiGLUGateF32(
            [projected],
            grid: (total, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[B, T, intermediateSize]],
            outputDTypes: [projected.dtype]
        )[0]
    }
}
