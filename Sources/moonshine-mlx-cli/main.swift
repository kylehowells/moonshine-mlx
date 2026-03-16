import ArgumentParser
import Foundation
import MLX
import MoonshineMLX

@main
struct MoonshineCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "moonshine-mlx",
        abstract: "Moonshine speech-to-text powered by MLX on Apple Silicon.",
        version: "0.1.0"
    )

    @Argument(help: "Audio file(s) to transcribe.")
    var audioFiles: [String]

    @Option(name: .long, help: "Model: HuggingFace repo ID or local path.")
    var model: String = "UsefulSensors/moonshine-streaming-tiny"

    @Option(name: .long, help: "HuggingFace token for gated models.")
    var hfToken: String?

    @Flag(name: .long, help: "Enable streaming mode (process audio in chunks).")
    var stream: Bool = false

    @Option(name: .long, help: "Streaming chunk size in milliseconds.")
    var chunkMs: Int = 500

    @Flag(name: .long, help: "Enable word-level timestamps.")
    var wordTimestamps: Bool = false

    @Option(name: .long, help: "Sampling temperature (0 = greedy).")
    var temperature: Float = 0.0

    @Option(name: .long, help: "Maximum tokens to generate per segment.")
    var maxTokens: Int = 500

    @Flag(name: .long, help: "Print verbose timing information.")
    var verbose: Bool = false

    @Flag(name: .long, help: "Output JSON metrics (for benchmarking).")
    var json: Bool = false

    func run() throws {
        // Load model
        let loadStart = CFAbsoluteTimeGetCurrent()
        if verbose {
            printErr("Loading model: \(model)...")
        }
        let moonshine = try MoonshineModel.load(from: model, hfToken: hfToken)
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart

        if verbose {
            printErr(String(format: "Model loaded in %.2fs", loadTime))
            printErr("Sample rate: \(moonshine.sampleRate) Hz")
            printErr("")
        }

        for audioPath in audioFiles {
            let url = URL(fileURLWithPath: audioPath)

            if verbose {
                printErr("Transcribing: \(url.lastPathComponent)")
            }

            // Load audio
            let (_, audio) = try AudioIO.loadMono(url: url, targetSampleRate: moonshine.sampleRate)
            let duration = Double(audio.dim(0)) / Double(moonshine.sampleRate)

            if verbose {
                printErr(String(format: "Audio: %.1fs, %d samples", duration, audio.dim(0)))
            }

            if stream {
                try transcribeStreaming(moonshine: moonshine, audio: audio, duration: duration)
            } else if wordTimestamps {
                transcribeWithTimestamps(moonshine: moonshine, audio: audio, duration: duration)
            } else if json {
                transcribeJSON(moonshine: moonshine, audio: audio, duration: duration,
                               audioPath: audioPath, loadTime: loadTime)
            } else {
                transcribeOffline(moonshine: moonshine, audio: audio, duration: duration)
            }

            if !json {
                print()
            }
        }
    }

    // MARK: - Offline Transcription

    private func transcribeOffline(moonshine: MoonshineModel, audio: MLXArray, duration: Double) {
        let result = moonshine.generate(
            audio: audio,
            maxTokens: maxTokens,
            temperature: temperature
        )

        print(result.text)

        if verbose {
            print(String(format: "\n  Time: %.2fs | Tokens: %d | Speed: %.1f tok/s | RTF: %.2fx",
                         result.totalTime, result.generationTokens,
                         result.tokensPerSecond, result.totalTime / duration))
        }
    }

    // MARK: - JSON Metrics (for benchmarking)

    private func transcribeJSON(moonshine: MoonshineModel, audio: MLXArray, duration: Double,
                                 audioPath: String, loadTime: Double) {
        Memory.peakMemory = 0  // reset

        let result = moonshine.generate(
            audio: audio,
            maxTokens: maxTokens,
            temperature: temperature
        )
        eval()
        let peakMem = Memory.peakMemory

        let metrics: [String: Any] = [
            "text": result.text,
            "audio_duration_s": round(duration * 1000) / 1000,
            "audio_file": URL(fileURLWithPath: audioPath).lastPathComponent,
            "model_load_time_s": round(loadTime * 10000) / 10000,
            "transcription_time_s": round(result.totalTime * 10000) / 10000,
            "generation_tokens": result.generationTokens,
            "tokens_per_second": round(result.tokensPerSecond * 100) / 100,
            "real_time_factor": duration > 0 ? round((result.totalTime / duration) * 10000) / 10000 : 0,
            "peak_memory_bytes": peakMem,
            "peak_memory_mb": round(Double(peakMem) / 1e6 * 10) / 10,
        ]

        if let data = try? JSONSerialization.data(
            withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys]),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }

    // MARK: - Word Timestamps

    private func transcribeWithTimestamps(moonshine: MoonshineModel, audio: MLXArray, duration: Double) {
        let (result, words) = moonshine.generateWithWordTimestamps(
            audio: audio,
            maxTokens: maxTokens
        )

        print(result.text)
        print()

        for word in words {
            print(String(format: "  [%6.2f - %6.2f] (%.2f) %@",
                         word.start, word.end, word.probability, word.word))
        }

        if verbose {
            print(String(format: "\n  Time: %.2fs | Tokens: %d | Speed: %.1f tok/s",
                         result.totalTime, result.generationTokens, result.tokensPerSecond))
        }
    }

    // MARK: - Streaming Transcription

    private func transcribeStreaming(moonshine: MoonshineModel, audio: MLXArray, duration: Double) throws {
        let state = moonshine.createStream()
        moonshine.startStream(state)

        let chunkSamples = moonshine.sampleRate * chunkMs / 1000
        let totalSamples = audio.dim(0)
        var offset = 0
        var lastText = ""
        let streamStart = CFAbsoluteTimeGetCurrent()

        while offset < totalSamples {
            let end = min(offset + chunkSamples, totalSamples)
            let chunk = audio[offset ..< end]
            let isFinal = end >= totalSamples

            moonshine.addAudio(state, chunk: chunk)
            let text = moonshine.transcribe(state, isFinal: isFinal, maxTokens: maxTokens, temperature: temperature)

            if !text.isEmpty && text != lastText {
                print("\r\u{1B}[K\(text)", terminator: isFinal ? "\n" : "")
                fflush(stdout)
                lastText = text
            }

            offset = end
        }

        moonshine.stopStream(state)

        if verbose {
            let elapsed = CFAbsoluteTimeGetCurrent() - streamStart
            printErr(String(format: "\n  Streaming time: %.2fs | RTF: %.2fx", elapsed, elapsed / duration))
        }
    }

    // MARK: - Helpers

    private func printErr(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}
