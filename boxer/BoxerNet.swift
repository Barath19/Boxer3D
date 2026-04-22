// BoxerNet.swift
// ONNX Runtime inference wrapper for BoxerNet on iOS.
//
// Dependencies (add to Package.swift or Podfile):
//   - onnxruntime-objc (via SPM: https://github.com/microsoft/onnxruntime-swift-package-manager)
//
// Usage:
//   let boxer = try BoxerNet(modelPath: Bundle.main.path(forResource: "BoxerNet", ofType: "onnx")!)
//   let results = try boxer.predict(
//       pixelBuffer: frame.capturedImage,
//       depthMap: frame.sceneDepth!.depthMap,
//       cameraIntrinsics: frame.camera.intrinsics,
//       cameraTransform: frame.camera.transform,
//       boxes2D: yoloDetections
//   )

import Foundation
import Accelerate
import simd
import OnnxRuntimeBindings

// MARK: - Data Types

/// A single 3D bounding box detection.
struct Detection3D {
    /// Centre position in ARKit world coordinates (metres).
    let center: simd_float3
    /// Box dimensions (width, height, depth) in metres.
    let size: simd_float3
    /// Yaw angle in radians [-pi/2, pi/2].
    let yaw: Float
    /// Detection confidence [0, 1].
    let confidence: Float
    /// Full 4x4 transform for placing in ARKit scene.
    let worldTransform: simd_float4x4
    /// Object class label from YOLO.
    let label: String?
}

/// A 2D bounding box from YOLO in pixel coordinates.
struct Box2D {
    let xmin: Float
    let ymin: Float
    let xmax: Float
    let ymax: Float
    var label: String? = nil
    var score: Float = 0
}

// MARK: - BoxerNet

final class BoxerNet {
    private let session: ORTSession
    private let env: ORTEnv

    /// Image size the model expects (960x960).
    static let imageSize: Int = 960
    /// Patch size used by DINOv3.
    static let patchSize: Int = 16
    /// Feature grid dimensions.
    static let gridH: Int = imageSize / patchSize  // 60
    static let gridW: Int = imageSize / patchSize  // 60
    static let numPatches: Int = gridH * gridW     // 3600

    init(modelPath: String, useCoreML: Bool = false) throws {
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        if useCoreML {
            // CoreML EP can spike memory during BoxerNet compilation on device.
            let coreMLOpts = ORTCoreMLExecutionProviderOptions()
            try opts.appendCoreMLExecutionProvider(with: coreMLOpts)
        }
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: opts)
    }

    // MARK: - Public API

    /// Run full pipeline: preprocess, infer, postprocess.
    ///
    /// - Parameters:
    ///   - image: RGB pixel data resized to 960x960, float32 in [0, 1], shape (3, 960, 960) in CHW.
    ///   - depthMap: LiDAR depth map from ARKit (metres). Can be smaller than 960x960.
    ///   - intrinsics: 3x3 camera intrinsics matrix from ARFrame.camera.intrinsics.
    ///   - cameraTransform: 4x4 world transform from ARFrame.camera.transform.
    ///   - boxes2D: Array of YOLO 2D detections in pixel coords (for the 960x960 image).
    ///   - confidenceThreshold: Minimum confidence to keep a detection.
    /// - Returns: Array of 3D detections in world coordinates.
    func predict(
        image: [Float],            // CHW flat array, len = 3 * 960 * 960
        depthMap: [[Float]],       // HxW depth in metres (0 = invalid)
        intrinsics: simd_float3x3, // fx, fy, cx, cy from ARKit
        cameraTransform: simd_float4x4,
        boxes2D: [Box2D],
        confidenceThreshold: Float = 0.3
    ) throws -> [Detection3D] {
        guard !boxes2D.isEmpty else { return [] }

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        // 1. Convert ARKit camera (OpenGL: -Z forward, Y up) to
        //    OpenCV convention (Z forward, Y down) that BoxerNet expects.
        //    Flip Y and Z axes of the camera local frame.
        let flipYZ = simd_float4x4(columns: (
            simd_float4( 1,  0,  0, 0),
            simd_float4( 0, -1,  0, 0),
            simd_float4( 0,  0, -1, 0),
            simd_float4( 0,  0,  0, 1)
        ))
        let T_wc = cameraTransform * flipYZ

        // Gravity-aligned voxel frame.
        let T_wv = gravityAlign(T_worldCam: T_wc)
        let T_vc = T_wv.inverse * T_wc

        // 2. Build SDP patches (median LiDAR depth per 16x16 patch).
        let sdpPatches = buildSDPPatches(
            depthMap: depthMap,
            fx: fx, fy: fy, cx: cx, cy: cy
        )

        // 3. Normalise 2D boxes.
        let W = Float(Self.imageSize)
        let H = Float(Self.imageSize)
        var bb2dFlat: [Float] = []
        for box in boxes2D {
            bb2dFlat.append((box.xmin + 0.5) / W)
            bb2dFlat.append((box.xmax + 0.5) / W)
            bb2dFlat.append((box.ymin + 0.5) / H)
            bb2dFlat.append((box.ymax + 0.5) / H)
        }

        // 4. Compute Plucker ray encoding.
        let rayEncoding = buildRayEncoding(T_vc: T_vc, fx: fx, fy: fy, cx: cx, cy: cy)

        // 5. Run ONNX inference.
        let M = boxes2D.count
        let (centers, sizes, yaws, confidences) = try runInference(
            image: image,
            sdpPatches: sdpPatches,
            bb2d: bb2dFlat,
            rayEncoding: rayEncoding,
            numBoxes: M
        )

        // 6. Postprocess: voxel → world coords.
        var detections: [Detection3D] = []
        for i in 0..<M {
            let conf = confidences[i]
            guard conf >= confidenceThreshold else { continue }

            let centerVoxel = simd_float3(centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2])
            let size = simd_float3(sizes[i * 3], sizes[i * 3 + 1], sizes[i * 3 + 2])
            let yaw = yaws[i]

            // Transform centre: voxel → world.
            let centerWorld = (T_wv * simd_float4(centerVoxel, 1.0)).xyz

            // Build world rotation: T_world_voxel.R * R_yaw.
            let R_wv = upperLeft3x3(T_wv)
            let R_yaw = rotationZ(angle: yaw)
            let R_world = R_wv * R_yaw

            // Build 4x4 transform for ARKit placement.
            var transform = simd_float4x4(1.0)
            transform[0] = simd_float4(R_world[0], 0)
            transform[1] = simd_float4(R_world[1], 0)
            transform[2] = simd_float4(R_world[2], 0)
            transform[3] = simd_float4(centerWorld, 1)

            detections.append(Detection3D(
                center: centerWorld,
                size: size,
                yaw: yaw,
                confidence: conf,
                worldTransform: transform,
                label: boxes2D[i].label
            ))
        }

        return detections
    }

    // MARK: - ONNX Inference

    private func runInference(
        image: [Float],
        sdpPatches: [Float],
        bb2d: [Float],
        rayEncoding: [Float],
        numBoxes: Int
    ) throws -> (centers: [Float], sizes: [Float], yaws: [Float], confidences: [Float]) {
        let S = Self.imageSize
        let gH = Self.gridH
        let gW = Self.gridW
        let N = Self.numPatches

        // Create input tensors.
        let imageData = Data(bytes: image, count: image.count * MemoryLayout<Float>.stride)
        let imageTensor = try ORTValue(
            tensorData: NSMutableData(data: imageData),
            elementType: .float,
            shape: [1, 3, NSNumber(value: S), NSNumber(value: S)]
        )

        let sdpData = Data(bytes: sdpPatches, count: sdpPatches.count * MemoryLayout<Float>.stride)
        let sdpTensor = try ORTValue(
            tensorData: NSMutableData(data: sdpData),
            elementType: .float,
            shape: [1, 1, NSNumber(value: gH), NSNumber(value: gW)]
        )

        let bb2dData = Data(bytes: bb2d, count: bb2d.count * MemoryLayout<Float>.stride)
        let bb2dTensor = try ORTValue(
            tensorData: NSMutableData(data: bb2dData),
            elementType: .float,
            shape: [1, NSNumber(value: numBoxes), 4]
        )

        let rayData = Data(bytes: rayEncoding, count: rayEncoding.count * MemoryLayout<Float>.stride)
        let rayTensor = try ORTValue(
            tensorData: NSMutableData(data: rayData),
            elementType: .float,
            shape: [1, NSNumber(value: N), 6]
        )

        // Run.
        let outputs = try session.run(
            withInputs: [
                "image": imageTensor,
                "sdp_patches": sdpTensor,
                "bb2d": bb2dTensor,
                "ray_encoding": rayTensor,
            ],
            outputNames: ["center", "size", "yaw", "confidence"],
            runOptions: nil
        )

        // Extract outputs.
        let centersData = try outputs["center"]!.tensorData() as Data
        let sizesData = try outputs["size"]!.tensorData() as Data
        let yawsData = try outputs["yaw"]!.tensorData() as Data
        let confsData = try outputs["confidence"]!.tensorData() as Data

        let centers = centersData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let sizes = sizesData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let yaws = yawsData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let confidences = confsData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        return (centers, sizes, yaws, confidences)
    }

    // MARK: - Preprocessing: SDP Patches

    /// Project LiDAR depth to 960x960 image, compute median depth per 16x16 patch.
    private func buildSDPPatches(
        depthMap: [[Float]],
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> [Float] {
        let S = Self.imageSize
        let P = Self.patchSize
        let gH = Self.gridH
        let gW = Self.gridW

        // Accumulate depths per patch.
        var patchDepths = [[Float]](repeating: [], count: gH * gW)

        let depthH = depthMap.count
        guard depthH > 0 else {
            return [Float](repeating: -1.0, count: gH * gW)
        }
        let depthW = depthMap[0].count

        // Scale factors from depth map to 960x960.
        let scaleX = Float(S) / Float(depthW)
        let scaleY = Float(S) / Float(depthH)

        // Scale intrinsics to 960x960.
        let fxS = fx * scaleX
        let fyS = fy * scaleY
        let cxS = cx * scaleX
        let cyS = cy * scaleY

        // For each depth pixel, project to 960x960 and assign to a patch.
        let step = max(1, Int(sqrt(Float(depthH * depthW) / 20000.0)))
        for v in stride(from: 0, to: depthH, by: step) {
            for u in stride(from: 0, to: depthW, by: step) {
                let z = depthMap[v][u]
                guard z > 0 else { continue }

                // Pixel in 960x960 space.
                let px = Float(u) * scaleX
                let py = Float(v) * scaleY

                let pi = Int(py) / P
                let pj = Int(px) / P
                guard pi >= 0, pi < gH, pj >= 0, pj < gW else { continue }

                patchDepths[pi * gW + pj].append(z)
            }
        }

        // Compute median per patch.
        var result = [Float](repeating: -1.0, count: gH * gW)
        for idx in 0..<(gH * gW) {
            var depths = patchDepths[idx]
            guard !depths.isEmpty else { continue }
            depths.sort()
            result[idx] = depths[depths.count / 2]
        }
        return result
    }

    // MARK: - Preprocessing: Plucker Ray Encoding

    /// Compute 6D Plucker ray encoding for each patch centre.
    private func buildRayEncoding(
        T_vc: simd_float4x4,
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> [Float] {
        let P = Float(Self.patchSize)
        let gH = Self.gridH
        let gW = Self.gridW

        // Scale intrinsics to 960x960 (assuming fx/fy/cx/cy are for original camera res).
        // NOTE: Caller should provide intrinsics already scaled to 960x960.

        let R_vc = upperLeft3x3(T_vc)
        let originCam = simd_float3(0, 0, 0)
        let originVoxel = (T_vc * simd_float4(originCam, 1.0)).xyz

        var result = [Float](repeating: 0, count: gH * gW * 6)

        for i in 0..<gH {
            for j in 0..<gW {
                let u = Float(j) * P + P / 2.0
                let v = Float(i) * P + P / 2.0

                // Unproject to camera frame (pinhole).
                var dirCam = simd_float3(
                    (u - cx) / fx,
                    (v - cy) / fy,
                    1.0
                )
                dirCam = simd_normalize(dirCam)

                // Rotate to voxel frame.
                var dirVoxel = R_vc * dirCam
                dirVoxel = simd_normalize(dirVoxel)

                // Moment: m = origin x direction.
                let moment = simd_cross(originVoxel, dirVoxel)

                let idx = (i * gW + j) * 6
                result[idx + 0] = dirVoxel.x
                result[idx + 1] = dirVoxel.y
                result[idx + 2] = dirVoxel.z
                result[idx + 3] = moment.x
                result[idx + 4] = moment.y
                result[idx + 5] = moment.z
            }
        }
        return result
    }

    // MARK: - Preprocessing: Gravity Alignment

    /// Compute gravity-aligned voxel frame matching Python's
    /// `gravity_align_T_world_cam(T_wc, z_grav=True)`.
    ///
    /// - Parameter gravity_w: Gravity direction in world frame.
    ///   ARKit uses (0, -1, 0).  The original Aria VIO uses (0, 0, -1).
    private func gravityAlign(
        T_worldCam: simd_float4x4,
        gravity_w: simd_float3 = simd_float3(0, -1, 0)
    ) -> simd_float4x4 {
        let R_wc = upperLeft3x3(T_worldCam)
        let t_wc = simd_float3(T_worldCam[3].x, T_worldCam[3].y, T_worldCam[3].z)

        let g_w = simd_normalize(gravity_w)

        // Camera forward (col 2 of R_wc) in world, projected orthogonal to gravity.
        let camZ_w = R_wc * simd_float3(0, 0, 1)
        var d3 = camZ_w - g_w * simd_dot(camZ_w, g_w) // reject gravity component
        if simd_length(d3) < 1e-6 {
            d3 = d3 + simd_float3(0, 0.001, 0) // tiny offset to avoid degenerate cross
        }

        let d2 = simd_cross(d3, g_w)

        // R_world_cg: columns are [g_w, d2, d3] (gravity is X axis — Aria convention)
        var R_wcg = simd_float3x3(columns: (g_w, d2, d3))
        // Normalize columns
        R_wcg[0] = simd_normalize(R_wcg[0])
        R_wcg[1] = simd_normalize(R_wcg[1])
        R_wcg[2] = simd_normalize(R_wcg[2])

        // Extra rotation to make Z the gravity axis (z_grav=True).
        // Python: R_cg_cgz = [[0,-1,0],[0,0,1],[-1,0,0]]
        // R_world_cgz = R_world_cg @ R_cg_cgz.inverse()
        let R_cg_cgz = simd_float3x3(columns: (
            simd_float3( 0,  0, -1),
            simd_float3(-1,  0,  0),
            simd_float3( 0,  1,  0)
        ))
        let R_world_cgz = R_wcg * R_cg_cgz.inverse

        // Build 4x4.
        var T_wv = simd_float4x4(1.0)
        T_wv[0] = simd_float4(R_world_cgz[0], 0)
        T_wv[1] = simd_float4(R_world_cgz[1], 0)
        T_wv[2] = simd_float4(R_world_cgz[2], 0)
        T_wv[3] = simd_float4(t_wc, 1)
        return T_wv
    }
}

// MARK: - simd Helpers

private func upperLeft3x3(_ m: simd_float4x4) -> simd_float3x3 {
    return simd_float3x3(
        simd_float3(m[0].x, m[0].y, m[0].z),
        simd_float3(m[1].x, m[1].y, m[1].z),
        simd_float3(m[2].x, m[2].y, m[2].z)
    )
}

private func rotationZ(angle: Float) -> simd_float3x3 {
    let c = cos(angle)
    let s = sin(angle)
    return simd_float3x3(
        simd_float3(c, s, 0),
        simd_float3(-s, c, 0),
        simd_float3(0, 0, 1)
    )
}

private extension simd_float4 {
    var xyz: simd_float3 { simd_float3(x, y, z) }
}
