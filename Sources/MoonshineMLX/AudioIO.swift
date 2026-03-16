import AVFoundation
import Foundation
import MLX

public enum AudioIOError: Error, LocalizedError {
    case cannotCreateBuffer
    case cannotCreateFormat
    case cannotReadChannelData
    case cannotResample

    public var errorDescription: String? {
        switch self {
        case .cannotCreateBuffer: "Failed to allocate audio buffer."
        case .cannotCreateFormat: "Failed to create audio format."
        case .cannotReadChannelData: "Failed to read float channel data."
        case .cannotResample: "Audio resampling failed."
        }
    }
}

public enum AudioIO {
    /// Default sample rate for Moonshine models.
    public static let defaultSampleRate = 16000

    /// Load mono audio from a file, resampling to 16 kHz.
    public static func loadMono(url: URL, targetSampleRate: Int = defaultSampleRate) throws -> (sampleRate: Int, audio: MLXArray) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioIOError.cannotCreateBuffer
        }
        try file.read(into: buffer)

        guard let channels = buffer.floatChannelData else {
            throw AudioIOError.cannotReadChannelData
        }

        let samples = Array(UnsafeBufferPointer(start: channels[0], count: Int(buffer.frameLength)))
        let sr = Int(format.sampleRate)

        if targetSampleRate > 0 && targetSampleRate != sr {
            let resampled = try resample(samples, from: sr, to: targetSampleRate)
            return (targetSampleRate, MLXArray(resampled))
        }
        return (sr, MLXArray(samples))
    }

    /// Get the duration of an audio file in seconds.
    public static func duration(url: URL) throws -> Double {
        let file = try AVAudioFile(forReading: url)
        return Double(file.length) / file.processingFormat.sampleRate
    }

    /// Resample audio from one sample rate to another.
    public static func resample(_ input: [Float], from sourceSR: Int, to targetSR: Int) throws -> [Float] {
        if input.isEmpty || sourceSR == targetSR { return input }

        guard let inFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sourceSR), channels: 1, interleaved: false
        ), let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(targetSR), channels: 1, interleaved: false
        ), let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw AudioIOError.cannotResample
        }

        let inFrameCount = AVAudioFrameCount(input.count)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: inFrameCount) else {
            throw AudioIOError.cannotCreateBuffer
        }
        inputBuffer.frameLength = inFrameCount
        input.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            memcpy(inputBuffer.floatChannelData![0], base, input.count * MemoryLayout<Float>.size)
        }

        let ratio = Double(targetSR) / Double(sourceSR)
        let estimatedFrames = max(1, Int(ceil(Double(input.count) * ratio)) + 64)
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outFormat, frameCapacity: AVAudioFrameCount(estimatedFrames)
        ) else {
            throw AudioIOError.cannotCreateBuffer
        }

        final class InputProvider: @unchecked Sendable {
            let buffer: AVAudioPCMBuffer
            var consumed = false
            init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
        }

        let provider = InputProvider(inputBuffer)
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if provider.consumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            provider.consumed = true
            outStatus.pointee = .haveData
            return provider.buffer
        }

        if conversionError != nil {
            throw AudioIOError.cannotResample
        }
        guard status == .haveData || status == .endOfStream || status == .inputRanDry else {
            throw AudioIOError.cannotResample
        }

        let outCount = Int(outputBuffer.frameLength)
        guard let outData = outputBuffer.floatChannelData?[0], outCount > 0 else {
            return []
        }
        return Array(UnsafeBufferPointer(start: outData, count: outCount))
    }
}
