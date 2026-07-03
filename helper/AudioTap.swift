// Captures system audio via ScreenCaptureKit, writes 16kHz mono Float32 PCM to stdout.
// Build: helper/build.sh   Run: helper/audiotap > out.raw
import Foundation
import CoreMedia
import ScreenCaptureKit

final class AudioOutput: NSObject, SCStreamOutput, SCStreamDelegate {
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio,
              let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        var length = 0
        var dataPointer: UnsafeMutablePointer<CChar>?
        let status = CMBlockBufferGetDataPointer(
            blockBuffer, atOffset: 0, lengthAtOffsetOut: nil,
            totalLengthOut: &length, dataPointerOut: &dataPointer)
        guard status == kCMBlockBufferNoErr, let ptr = dataPointer else { return }
        FileHandle.standardOutput.write(Data(bytes: ptr, count: length))
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("stream stopped: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

let semaphore = DispatchSemaphore(value: 0)
Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            FileHandle.standardError.write("no display\n".data(using: .utf8)!)
            exit(1)
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        // SCStream requires a video config even for audio-only use; keep it tiny.
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let output = AudioOutput()
        let stream = SCStream(filter: filter, configuration: config, delegate: output)
        try stream.addStreamOutput(output, type: .audio,
                                   sampleHandlerQueue: DispatchQueue(label: "audio"))
        try await stream.startCapture()
        FileHandle.standardError.write("capturing\n".data(using: .utf8)!)
    } catch {
        FileHandle.standardError.write("failed: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}
semaphore.wait()
