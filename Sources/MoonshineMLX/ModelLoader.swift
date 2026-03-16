import Dispatch
import Foundation
import Hub
import MLX
import MLXNN

// MARK: - Model Loader

public enum MoonshineModelLoader {
    /// Default HuggingFace model repo for the tiny streaming model.
    public static let defaultModelRepo = "UsefulSensors/moonshine-streaming-tiny"

    // MARK: - Public API

    /// Load a Moonshine model from a HuggingFace repo ID or local path.
    public static func load(
        from source: String = defaultModelRepo,
        hfToken: String? = nil
    ) throws -> MoonshineModel {
        let modelDir: URL

        // Check if source is a local path
        let localURL = URL(fileURLWithPath: source)
        if FileManager.default.fileExists(atPath: localURL.path) {
            modelDir = localURL
        } else {
            // Try as HuggingFace repo ID
            modelDir = try downloadFromHub(repoID: source, hfToken: hfToken)
        }

        return try loadFromDirectory(modelDir)
    }

    /// Load a Moonshine model from a local directory.
    public static func loadFromDirectory(_ directory: URL) throws -> MoonshineModel {
        // Load config
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        guard let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw MoonshineError.invalidConfig("Failed to parse config.json")
        }
        let config = MoonshineModelConfig.from(dict: configDict)

        // Create model
        let model = MoonshineModel(config: config)

        // Apply quantization if configured
        if let quantConfig = config.quantization {
            applyQuantization(model: model, groupSize: quantConfig.groupSize, bits: quantConfig.bits)
        }

        // Load weights
        let weightsURL = try findWeightsFile(in: directory)
        let weights = try MLX.loadArrays(url: weightsURL)
        let sanitized = sanitizeWeights(weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized))
        eval(model.parameters())

        // Pre-compute offset weights for UnitOffsetLayerNorm layers
        model.prepareForInference()

        // Load tokenizer
        let tokenizer = try runBlocking {
            try await MoonshineTokenizer.load(from: directory)
        }
        model.tokenizer = tokenizer


        return model
    }

    // MARK: - Weight Handling

    /// Sanitize weight keys from HuggingFace format to MLX module format.
    static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let needsHFMapping = weights.keys.contains { $0.hasPrefix("model.") }
        var out: [String: MLXArray] = [:]

        for (key, var val) in weights {
            var k = key

            if needsHFMapping {
                // Strip "model." prefix
                if k.hasPrefix("model.") {
                    k = String(k.dropFirst(6))
                }

                // Encoder norms: .gamma -> .weight (UnitOffsetLayerNorm)
                k = k.replacingOccurrences(of: ".gamma", with: ".weight")

                // Conv1d weights: PyTorch [out, in, k] -> MLX [out, k, in]
                if k.contains("conv") && k.hasSuffix(".weight") && val.ndim == 3 {
                    val = val.transposed(0, 2, 1)
                }
            }

            out[k] = val
        }

        return out
    }

    /// Apply quantization to Linear layers whose dimensions are compatible.
    static func applyQuantization(model: MoonshineModel, groupSize: Int, bits: Int) {
        quantize(model: model, groupSize: groupSize, bits: bits) { _, module in
            guard let linear = module as? Linear else { return false }
            // Skip layers where the weight's last dimension isn't divisible by group size
            // (e.g. the embedder's Linear(80, 320) where 80 % 64 != 0)
            return linear.weight.dim(-1) % groupSize == 0
        }
    }

    // MARK: - File Resolution

    static func findWeightsFile(in directory: URL) throws -> URL {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensors = contents.filter { $0.pathExtension == "safetensors" }

        // Prefer model.safetensors
        if let model = safetensors.first(where: { $0.lastPathComponent == "model.safetensors" }) {
            return model
        }
        // Fall back to any safetensors file
        if let first = safetensors.first {
            return first
        }

        throw MoonshineError.missingWeights(directory.path)
    }

    // MARK: - Hub Download

    static func downloadFromHub(repoID: String, hfToken: String? = nil) throws -> URL {
        let hub = HubApi(hfToken: hfToken)
        return try runBlocking {
            try await hub.snapshot(from: repoID, matching: [
                "*.safetensors", "*.json", "*.txt", "*.model",
            ])
        }
    }

    // MARK: - Async Bridge

    private final class ResultBox<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<Value, Error>?

        init() {}

        func set(_ value: Result<Value, Error>) {
            lock.lock()
            result = value
            lock.unlock()
        }

        func get() -> Result<Value, Error>? {
            lock.lock()
            defer { lock.unlock() }
            return result
        }
    }

    static func runBlocking<T>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
        let box = ResultBox<T>()
        let semaphore = DispatchSemaphore(value: 0)

        Task.detached(priority: .userInitiated) {
            do {
                box.set(.success(try await operation()))
            } catch {
                box.set(.failure(error))
            }
            semaphore.signal()
        }

        semaphore.wait()
        guard let result = box.get() else {
            throw MoonshineError.downloadFailed("Hub download produced no result.")
        }
        return try result.get()
    }
}

// MARK: - Errors

public enum MoonshineError: Error, LocalizedError {
    case invalidConfig(String)
    case missingWeights(String)
    case downloadFailed(String)
    case noTokenizer
    case audioLoadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfig(let msg): "Invalid config: \(msg)"
        case .missingWeights(let path): "No safetensors weights found in: \(path)"
        case .downloadFailed(let msg): "Download failed: \(msg)"
        case .noTokenizer: "Tokenizer not loaded."
        case .audioLoadFailed(let msg): "Audio load failed: \(msg)"
        }
    }
}
