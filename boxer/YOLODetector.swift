import Foundation
import OnnxRuntimeBindings

/// YOLO detection result.
struct YOLOBox {
    let xmin: Float
    let ymin: Float
    let xmax: Float
    let ymax: Float
    let label: String
    let score: Float
}

/// YOLO11n ONNX inference wrapper.
final class YOLODetector {
    private let session: ORTSession
    private let env: ORTEnv

    /// COCO class names.
    static let classNames: [String] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]

    init(modelPath: String, useCoreML: Bool = true) throws {
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        if useCoreML {
            let coreMLOpts = ORTCoreMLExecutionProviderOptions()
            try opts.appendCoreMLExecutionProvider(with: coreMLOpts)
        }
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: opts)
    }

    /// Run YOLO detection.
    /// - Parameters:
    ///   - image: CHW float32 array in [0, 1], length = 3 * 640 * 640.
    ///   - imageWidth: 640
    ///   - imageHeight: 640
    ///   - confThreshold: Minimum confidence.
    ///   - iouThreshold: NMS IoU threshold.
    /// - Returns: Array of 2D bounding boxes.
    func detect(
        image: [Float],
        imageWidth: Int = 640,
        imageHeight: Int = 640,
        confThreshold: Float = 0.25,
        iouThreshold: Float = 0.45
    ) throws -> [YOLOBox] {
        let imageData = Data(bytes: image, count: image.count * MemoryLayout<Float>.stride)
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: imageData),
            elementType: .float,
            shape: [1, 3, NSNumber(value: imageHeight), NSNumber(value: imageWidth)]
        )

        let outputs = try session.run(
            withInputs: ["images": inputTensor],
            outputNames: ["output0"],
            runOptions: nil
        )

        // YOLO11 output: (1, 84, 8400) — 4 box coords + 80 class scores per anchor.
        let outputData = try outputs["output0"]!.tensorData() as Data
        let values = outputData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        let numClasses = 80
        let numAnchors = 8400
        let stride = numAnchors

        var boxes: [YOLOBox] = []

        for a in 0..<numAnchors {
            // Box: cx, cy, w, h (indices 0-3).
            let cx = values[0 * stride + a]
            let cy = values[1 * stride + a]
            let w  = values[2 * stride + a]
            let h  = values[3 * stride + a]

            // Find best class.
            var bestClass = 0
            var bestScore: Float = 0
            for c in 0..<numClasses {
                let score = values[(4 + c) * stride + a]
                if score > bestScore {
                    bestScore = score
                    bestClass = c
                }
            }

            guard bestScore >= confThreshold else { continue }

            let xmin = cx - w / 2
            let ymin = cy - h / 2
            let xmax = cx + w / 2
            let ymax = cy + h / 2

            let label = bestClass < Self.classNames.count
                ? Self.classNames[bestClass]
                : "class_\(bestClass)"

            boxes.append(YOLOBox(
                xmin: xmin, ymin: ymin, xmax: xmax, ymax: ymax,
                label: label, score: bestScore
            ))
        }

        // NMS.
        boxes = nms(boxes: boxes, iouThreshold: iouThreshold)

        return boxes
    }

    /// Non-Maximum Suppression.
    private func nms(boxes: [YOLOBox], iouThreshold: Float) -> [YOLOBox] {
        let sorted = boxes.sorted { $0.score > $1.score }
        var keep: [YOLOBox] = []

        for box in sorted {
            var shouldKeep = true
            for kept in keep {
                if iou(box, kept) > iouThreshold && box.label == kept.label {
                    shouldKeep = false
                    break
                }
            }
            if shouldKeep {
                keep.append(box)
            }
        }
        return keep
    }

    private func iou(_ a: YOLOBox, _ b: YOLOBox) -> Float {
        let x1 = max(a.xmin, b.xmin)
        let y1 = max(a.ymin, b.ymin)
        let x2 = min(a.xmax, b.xmax)
        let y2 = min(a.ymax, b.ymax)
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin)
        let areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin)
        let union = areaA + areaB - intersection
        return union > 0 ? intersection / union : 0
    }
}
