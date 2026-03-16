import Foundation
import MLX
import MoonshineMLX

let model = try MoonshineModel.load(from: "UsefulSensors/moonshine-streaming-tiny")

for file in ["/tmp/bench_30s.wav", "/tmp/bench_60s.wav", "/tmp/bench_120s.wav"] {
    let (_, audio) = try AudioIO.loadMono(url: URL(fileURLWithPath: file), targetSampleRate: model.sampleRate)
    let dur = Double(audio.dim(0)) / Double(model.sampleRate)
    let stderr = FileHandle.standardError

    stderr.write(Data("\n=== \(file) (\(Int(dur))s) ===\n".utf8))

    // Direct generate
    let gen = model.generate(audio: audio)
    stderr.write(Data("generate(): \(gen.generationTokens) tok, \(String(format: "%.2f", gen.totalTime))s\n".utf8))
    stderr.write(Data("  text[:120]: \(String(gen.text.prefix(120)))\n".utf8))
    let endText = gen.text.count > 200 ? String(gen.text.suffix(80)) : ""
    stderr.write(Data("  text end: ...\(endText)\n".utf8))

    // Long generate
    let long = model.generateLong(audio: audio)
    stderr.write(Data("generateLong(): \(long.generationTokens) segments, \(String(format: "%.2f", long.totalTime))s\n".utf8))
    stderr.write(Data("  text[:120]: \(String(long.text.prefix(120)))\n".utf8))
    let longEnd = long.text.count > 200 ? String(long.text.suffix(80)) : ""
    stderr.write(Data("  text end: ...\(longEnd)\n".utf8))
}
