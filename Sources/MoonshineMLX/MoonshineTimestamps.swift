import Foundation
import MLX

/// Duration of each encoder output frame (50 Hz = 20 ms).
let frameDurationSeconds: Double = 0.02

// MARK: - Word Timing

public struct WordTiming: Sendable {
    public let word: String
    public let tokens: [Int]
    public let start: Double
    public let end: Double
    public let probability: Double
}

// MARK: - Dynamic Time Warping

/// DTW on a cost matrix. Returns (textIndices, timeIndices) as the alignment path.
func dtw(_ costMatrix: [[Float]]) -> (textIndices: [Int], timeIndices: [Int]) {
    let N = costMatrix.count
    guard N > 0 else { return ([], []) }
    let M = costMatrix[0].count
    guard M > 0 else { return ([], []) }

    // Cost and trace matrices
    var cost = [[Float]](repeating: [Float](repeating: Float.infinity, count: M + 1), count: N + 1)
    var trace = [[Int]](repeating: [Int](repeating: -1, count: M + 1), count: N + 1)
    cost[0][0] = 0.0

    for j in 1 ... M {
        for i in 1 ... N {
            let c0 = cost[i - 1][j - 1]  // diagonal
            let c1 = cost[i - 1][j]      // vertical (skip text)
            let c2 = cost[i][j - 1]      // horizontal (skip time)
            let c: Float
            let t: Int
            if c0 <= c1 && c0 <= c2 {
                c = c0; t = 0
            } else if c1 <= c2 {
                c = c1; t = 1
            } else {
                c = c2; t = 2
            }
            cost[i][j] = costMatrix[i - 1][j - 1] + c
            trace[i][j] = t
        }
    }

    // Backtrace
    var i = N, j = M
    for col in 0 ... M { trace[0][col] = 2 }
    for row in 0 ... N { trace[row][0] = 1 }
    var path: [(Int, Int)] = []
    while i > 0 || j > 0 {
        path.append((i - 1, j - 1))
        let t = trace[i][j]
        if t == 0 { i -= 1; j -= 1 }
        else if t == 1 { i -= 1 }
        else { j -= 1 }
    }
    path.reverse()

    return (path.map(\.0), path.map(\.1))
}

/// Simple 1D median filter.
func medianFilter1D(_ data: [Float], width: Int) -> [Float] {
    guard width > 1, data.count > 0 else { return data }
    let half = width / 2
    var out = [Float](repeating: 0, count: data.count)
    for i in 0 ..< data.count {
        let lo = max(0, i - half)
        let hi = min(data.count - 1, i + half)
        var window = Array(data[lo ... hi])
        window.sort()
        out[i] = window[window.count / 2]
    }
    return out
}

// MARK: - Alignment Extraction

/// Build word-level timestamps from per-step cross-attention weights.
///
/// - Parameters:
///   - crossQKPerStep: Outer list: one per decode step. Inner list: per layer. Each: [B, H, 1, S].
///   - tokens: Decoded token IDs (excluding BOS).
///   - tokenizer: Tokenizer for decoding and word boundary detection.
///   - numFrames: Number of encoder memory frames.
///   - timeOffset: Time offset added to all timestamps.
///   - medfiltWidth: Width of median filter for smoothing.
///   - tokenProbs: Per-token generation probabilities.
public func findAlignment(
    crossQKPerStep: [[[MLXArray]]],
    tokens: [Int],
    tokenToString: (Int) -> String?,
    decodeTokens: ([Int]) -> String,
    numFrames: Int,
    timeOffset: Double = 0.0,
    medfiltWidth: Int = 7,
    tokenProbs: [Float]? = nil
) -> [WordTiming] {
    guard !tokens.isEmpty, !crossQKPerStep.isEmpty else { return [] }

    // Build attention matrix: average across all layers and heads per step
    var attnRows: [[Float]] = []
    for stepQKs in crossQKPerStep {
        var stepAvg = [Float](repeating: 0, count: numFrames)
        var layerCount = 0
        for layerQKs in stepQKs {
            for layerQK in layerQKs {
                // layerQK: [B, H, 1, S]
                let w = softmax(layerQK[0, 0..., 0, 0...], axis: -1)  // [H, S]
                let meanW = w.mean(axis: 0)  // [S]
                eval(meanW)
                let arr: [Float] = meanW.asArray(Float.self)
                for k in 0 ..< min(arr.count, numFrames) {
                    stepAvg[k] += arr[k]
                }
                layerCount += 1
            }
        }
        if layerCount > 0 {
            for k in 0 ..< numFrames {
                stepAvg[k] /= Float(layerCount)
            }
        }
        attnRows.append(stepAvg)
    }

    // Z-score normalize per row
    for i in 0 ..< attnRows.count {
        let row = attnRows[i]
        let mean = row.reduce(0, +) / Float(max(row.count, 1))
        let variance = row.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(max(row.count, 1))
        let std = sqrt(variance) + 1e-6
        attnRows[i] = row.map { ($0 - mean) / std }
    }

    // Median filter per row
    if medfiltWidth > 1 {
        for i in 0 ..< attnRows.count {
            attnRows[i] = medianFilter1D(attnRows[i], width: medfiltWidth)
        }
    }

    // DTW on negative attention
    let negated = attnRows.map { $0.map { -$0 } }
    let (textIndices, timeIndices) = dtw(negated)

    let probs = tokenProbs ?? [Float](repeating: 1.0, count: tokens.count)

    // Group tokens into words using sentencepiece ▁ marker
    var words: [(text: String, tokenIndices: [Int])] = []
    var currentWordTokens: [Int] = []
    var currentTextTokens: [Int] = []

    for (i, tokId) in tokens.enumerated() {
        let piece = tokenToString(tokId) ?? ""
        if piece.hasPrefix("\u{2581}") && !currentWordTokens.isEmpty {
            let wordText = decodeTokens(currentTextTokens)
            words.append((wordText, currentWordTokens))
            currentWordTokens = []
            currentTextTokens = []
        }
        currentWordTokens.append(i)
        currentTextTokens.append(tokId)
    }
    if !currentWordTokens.isEmpty {
        let wordText = decodeTokens(currentTextTokens)
        words.append((wordText, currentWordTokens))
    }

    // Map token indices to frame indices via DTW path
    var jumps = [Bool](repeating: false, count: textIndices.count)
    jumps[0] = true
    for i in 1 ..< textIndices.count {
        jumps[i] = textIndices[i] != textIndices[i - 1]
    }
    var jumpTimes: [Int] = []
    for i in 0 ..< jumps.count where jumps[i] {
        jumpTimes.append(timeIndices[i])
    }

    // Build word timings
    var result: [WordTiming] = []
    for (wordText, wordTokIndices) in words {
        guard !wordTokIndices.isEmpty else { continue }
        let firstTok = wordTokIndices.first!
        let lastTok = wordTokIndices.last!

        let startFrame: Int
        if firstTok < jumpTimes.count {
            startFrame = jumpTimes[firstTok]
        } else {
            startFrame = jumpTimes.last ?? 0
        }

        let endFrame: Int
        if lastTok + 1 < jumpTimes.count {
            endFrame = jumpTimes[lastTok + 1]
        } else {
            endFrame = jumpTimes.last ?? numFrames
        }

        let startTime = timeOffset + Double(startFrame) * frameDurationSeconds
        let endTime = timeOffset + Double(endFrame) * frameDurationSeconds

        let wordProbs = wordTokIndices.compactMap { i in
            i < probs.count ? probs[i] : nil
        }
        let avgProb = wordProbs.isEmpty ? 1.0 : Double(wordProbs.reduce(0, +)) / Double(wordProbs.count)

        result.append(WordTiming(
            word: wordText.trimmingCharacters(in: .whitespaces),
            tokens: wordTokIndices.map { tokens[$0] },
            start: (startTime * 1000).rounded() / 1000,
            end: (endTime * 1000).rounded() / 1000,
            probability: (avgProb * 10000).rounded() / 10000
        ))
    }

    // Fix overlapping word boundaries
    for i in 1 ..< result.count {
        let prev = result[i - 1]
        let curr = result[i]
        if prev.end > curr.start {
            let mid = ((prev.end + curr.start) / 2 * 1000).rounded() / 1000
            result[i - 1] = WordTiming(
                word: prev.word, tokens: prev.tokens,
                start: prev.start, end: mid, probability: prev.probability
            )
            result[i] = WordTiming(
                word: curr.word, tokens: curr.tokens,
                start: mid, end: curr.end, probability: curr.probability
            )
        }
    }

    return result
}
