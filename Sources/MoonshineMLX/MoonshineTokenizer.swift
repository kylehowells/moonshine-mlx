import Foundation
import Tokenizers

// MARK: - Moonshine Tokenizer Wrapper

/// Wraps a HuggingFace tokenizer for Moonshine's SentencePiece vocabulary.
public final class MoonshineTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Load tokenizer from a local model directory containing tokenizer.json.
    public static func load(from directory: URL) async throws -> MoonshineTokenizer {
        let tok = try await AutoTokenizer.from(modelFolder: directory)
        return MoonshineTokenizer(tokenizer: tok)
    }

    /// Decode a sequence of token IDs to text.
    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    /// Get the string representation of a single token ID (for word boundary detection).
    public func tokenToString(_ tokenId: Int) -> String? {
        tokenizer.convertIdToToken(tokenId)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }
}
