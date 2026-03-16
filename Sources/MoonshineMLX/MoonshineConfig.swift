import Foundation

// MARK: - Encoder Configuration

public struct EncoderConfig: Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var frameMs: Double
    public var sampleRate: Int
    public var maxPositionEmbeddings: Int
    public var slidingWindows: [[Int]]?

    public var frameLen: Int { Int(frameMs * Double(sampleRate) / 1000.0) }

    public init(
        hiddenSize: Int = 320,
        intermediateSize: Int = 1280,
        numHiddenLayers: Int = 6,
        numAttentionHeads: Int = 8,
        numKeyValueHeads: Int = 8,
        headDim: Int = 40,
        frameMs: Double = 5.0,
        sampleRate: Int = 16000,
        maxPositionEmbeddings: Int = 4096,
        slidingWindows: [[Int]]? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.frameMs = frameMs
        self.sampleRate = sampleRate
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.slidingWindows = slidingWindows
    }

    public static func from(dict d: [String: Any]) -> EncoderConfig {
        EncoderConfig(
            hiddenSize: d["hidden_size"] as? Int ?? 320,
            intermediateSize: d["intermediate_size"] as? Int ?? 1280,
            numHiddenLayers: d["num_hidden_layers"] as? Int ?? 6,
            numAttentionHeads: d["num_attention_heads"] as? Int ?? 8,
            numKeyValueHeads: d["num_key_value_heads"] as? Int ?? 8,
            headDim: d["head_dim"] as? Int ?? 40,
            frameMs: d["frame_ms"] as? Double ?? 5.0,
            sampleRate: d["sample_rate"] as? Int ?? 16000,
            maxPositionEmbeddings: d["max_position_embeddings"] as? Int ?? 4096,
            slidingWindows: d["sliding_windows"] as? [[Int]]
        )
    }
}

// MARK: - Model Configuration

public struct MoonshineModelConfig: Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var tieWordEmbeddings: Bool
    public var encoderHiddenSize: Int
    public var encoder: EncoderConfig
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var decoderStartTokenId: Int
    public var padTokenId: Int
    public var partialRotaryFactor: Double
    public var ropeTheta: Double
    public var maxTokensPerSecond: Double
    public var quantization: QuantizationConfig?

    public init(
        modelType: String = "moonshine_streaming",
        hiddenSize: Int = 320,
        intermediateSize: Int = 1280,
        numHiddenLayers: Int = 6,
        numAttentionHeads: Int = 8,
        numKeyValueHeads: Int = 8,
        headDim: Int = 40,
        vocabSize: Int = 32768,
        maxPositionEmbeddings: Int = 4096,
        tieWordEmbeddings: Bool = false,
        encoderHiddenSize: Int? = nil,
        encoder: EncoderConfig? = nil,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        decoderStartTokenId: Int = 1,
        padTokenId: Int = 0,
        partialRotaryFactor: Double = 0.8,
        ropeTheta: Double = 10000.0,
        maxTokensPerSecond: Double = 6.5,
        quantization: QuantizationConfig? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.tieWordEmbeddings = tieWordEmbeddings
        self.encoder = encoder ?? EncoderConfig(
            hiddenSize: encoderHiddenSize ?? hiddenSize,
            intermediateSize: (encoderHiddenSize ?? hiddenSize) * 4,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim
        )
        self.encoderHiddenSize = encoderHiddenSize ?? self.encoder.hiddenSize
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.decoderStartTokenId = decoderStartTokenId
        self.padTokenId = padTokenId
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.maxTokensPerSecond = maxTokensPerSecond
        self.quantization = quantization
    }

    public static func from(dict d: [String: Any]) -> MoonshineModelConfig {
        var encConfig: EncoderConfig?
        if let encDict = d["encoder_config"] as? [String: Any] {
            encConfig = EncoderConfig.from(dict: encDict)
        }

        var partialRotary = d["partial_rotary_factor"] as? Double ?? 0.8
        var ropeTheta = d["rope_theta"] as? Double ?? 10000.0
        if let ropeParams = d["rope_parameters"] as? [String: Any] {
            partialRotary = ropeParams["partial_rotary_factor"] as? Double ?? partialRotary
            ropeTheta = ropeParams["rope_theta"] as? Double ?? ropeTheta
        }

        var quantConfig: QuantizationConfig?
        if let qd = d["quantization"] as? [String: Any] {
            quantConfig = QuantizationConfig(
                groupSize: qd["group_size"] as? Int ?? 64,
                bits: qd["bits"] as? Int ?? 8
            )
        }

        return MoonshineModelConfig(
            modelType: d["model_type"] as? String ?? "moonshine_streaming",
            hiddenSize: d["hidden_size"] as? Int ?? 320,
            intermediateSize: d["intermediate_size"] as? Int ?? 1280,
            numHiddenLayers: d["num_hidden_layers"] as? Int ?? 6,
            numAttentionHeads: d["num_attention_heads"] as? Int ?? 8,
            numKeyValueHeads: d["num_key_value_heads"] as? Int ?? 8,
            headDim: d["head_dim"] as? Int ?? 40,
            vocabSize: d["vocab_size"] as? Int ?? 32768,
            maxPositionEmbeddings: d["max_position_embeddings"] as? Int ?? 4096,
            tieWordEmbeddings: d["tie_word_embeddings"] as? Bool ?? false,
            encoderHiddenSize: d["encoder_hidden_size"] as? Int,
            encoder: encConfig,
            bosTokenId: d["bos_token_id"] as? Int ?? 1,
            eosTokenId: d["eos_token_id"] as? Int ?? 2,
            decoderStartTokenId: d["decoder_start_token_id"] as? Int ?? 1,
            padTokenId: d["pad_token_id"] as? Int ?? 0,
            partialRotaryFactor: partialRotary,
            ropeTheta: ropeTheta,
            maxTokensPerSecond: d["max_tokens_per_second"] as? Double ?? 6.5,
            quantization: quantConfig
        )
    }
}

// MARK: - Quantization Configuration

public struct QuantizationConfig: Sendable {
    public var groupSize: Int
    public var bits: Int

    public init(groupSize: Int = 64, bits: Int = 8) {
        self.groupSize = groupSize
        self.bits = bits
    }
}
