import SwiftUI
import ARKit
import SceneKit

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var viewModel: ARViewModel

    func makeUIView(context: Context) -> ARSCNView {
        let sceneView = ARSCNView()
        sceneView.autoenablesDefaultLighting = true
        sceneView.automaticallyUpdatesLighting = true

        guard ARWorldTrackingConfiguration.isSupported else {
            viewModel.configure(sceneView: sceneView, supportsSceneDepth: false)
            return sceneView
        }

        let config = ARWorldTrackingConfiguration()
        let supportsSceneDepth = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        if supportsSceneDepth {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.planeDetection = [.horizontal, .vertical]

        sceneView.session.run(config)
        viewModel.configure(sceneView: sceneView, supportsSceneDepth: supportsSceneDepth)

        return sceneView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
