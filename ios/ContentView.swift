//
//  ContentView.swift
//  Driver Monitoring System
//
//  SwiftUI screen showing camera preview, driver status, and telemetry bars.
//

import SwiftUI
import AVFoundation
import UIKit

// Custom UIView with native layer-backing for AVCaptureVideoPreviewLayer
public final class PreviewUIView: UIView {
    public override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }

    public var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        return layer as! AVCaptureVideoPreviewLayer
    }

    public override func layoutSubviews() {
        super.layoutSubviews()
        videoPreviewLayer.frame = bounds
        if #available(iOS 17.0, *) {
            if let connection = videoPreviewLayer.connection, connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        } else {
            if let connection = videoPreviewLayer.connection, connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }
    }
}

public struct CameraPreviewView: UIViewRepresentable {
    public let session: AVCaptureSession

    public init(session: AVCaptureSession) {
        self.session = session
    }

    public func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.backgroundColor = .black
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    public func updateUIView(_ uiView: PreviewUIView, context: Context) {
        if uiView.videoPreviewLayer.session != session {
            uiView.videoPreviewLayer.session = session
        }
    }
}

public struct ContentView: View {
    @StateObject private var vm = DMSViewModel()
    @State private var flashAlert = false

    public init() {}

    public var body: some View {
        ZStack {
            // 1. Live Front Camera View (Full Screen)
            CameraPreviewView(session: vm.cameraManager.captureSession)
                .ignoresSafeArea()

            // Subtle gradient vignette for high-contrast HUD visibility
            VStack {
                LinearGradient(
                    colors: [Color.black.opacity(0.75), Color.clear],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 140)

                Spacer()

                LinearGradient(
                    colors: [Color.clear, Color.black.opacity(0.8)],
                    startPoint: .top,
                    endPoint: .bottom
                )
                .frame(height: 240)
            }
            .ignoresSafeArea()

            // 2. Heads-Up Display (HUD) Overlay
            VStack {
                // Top Header Card
                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("EDGE DMS ACTIVE")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.cyan)

                        Text("STATE: \(vm.state.uppercased())")
                            .font(.system(size: 22, weight: .heavy, design: .rounded))
                            .foregroundColor(stateColor(vm.state))

                        Text("Confidence: \(Int(vm.confidence * 100))% • FPS: \(String(format: "%.1f", vm.fps))")
                            .font(.system(size: 12))
                            .foregroundColor(.white.opacity(0.85))
                    }

                    Spacer()

                    // Visual Alert Pulse Beacon
                    Circle()
                        .fill(stateColor(vm.state))
                        .frame(width: 18, height: 18)
                        .scaleEffect(vm.isAlertActive ? (flashAlert ? 1.4 : 0.9) : 1.0)
                        .animation(vm.isAlertActive ? Animation.easeInOut(duration: 0.3).repeatForever() : .default, value: flashAlert)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(Color.black.opacity(0.6))
                .cornerRadius(16)
                .padding(.horizontal)
                .padding(.top, 10)

                // Warning Flash Banner
                if vm.isAlertActive {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                        Text("WARNING: \(vm.state.uppercased()) DETECTED")
                            .fontWeight(.bold)
                    }
                    .font(.system(size: 14))
                    .foregroundColor(.white)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 20)
                    .background(Color.red.opacity(0.9))
                    .cornerRadius(20)
                    .transition(.scale)
                    .padding(.top, 4)
                }

                // Calibration / Status Banner
                if vm.isCalibrating || !vm.calibrationMessage.isEmpty {
                    Text(vm.calibrationMessage)
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.yellow)
                        .padding(.vertical, 6)
                        .padding(.horizontal, 14)
                        .background(Color.black.opacity(0.85))
                        .cornerRadius(12)
                        .padding(.top, 6)
                        .multilineTextAlignment(.center)
                }

                Spacer()

                // Face Searching Badge if driver not in frame
                if !vm.faceDetected {
                    Text("SEARCHING FOR DRIVER...")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundColor(.orange)
                        .padding(.vertical, 6)
                        .padding(.horizontal, 16)
                        .background(Color.black.opacity(0.75))
                        .cornerRadius(12)
                }

                Spacer()

                // Bottom Telemetry Card
                VStack(spacing: 10) {
                    HStack {
                        Text("REAL-TIME TELEMETRY")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.cyan)
                        Spacer()

                        // Quick 1-Tap Road Heading Zeroing
                        Button(action: {
                            vm.setRoadPoseBaseline()
                        }) {
                            HStack(spacing: 3) {
                                Image(systemName: "car.fill")
                                Text("Zero Heading")
                            }
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.black)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.orange)
                            .cornerRadius(8)
                        }

                        // Full Calibration (Eyes + Pose + Mouth Baseline)
                        Button(action: {
                            vm.startCalibration()
                        }) {
                            HStack(spacing: 3) {
                                Image(systemName: "scope")
                                Text("Calibrate All")
                            }
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.black)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.cyan)
                            .cornerRadius(8)
                        }
                    }

                    // EAR Progress Bar (Warn below threshold) with nudge controls
                    TelemetryBar(
                        title: "EAR (Eyes)",
                        value: vm.ear,
                        threshold: vm.earThreshold,
                        maxValue: 0.45,
                        warnBelow: true,
                        onDecrease: { vm.adjustEARThreshold(delta: -0.02) },
                        onIncrease: { vm.adjustEARThreshold(delta: +0.02) }
                    )

                    // MAR Progress Bar (Warn above threshold) with nudge controls
                    TelemetryBar(
                        title: "MAR (Mouth)",
                        value: vm.mar,
                        threshold: vm.marThreshold,
                        maxValue: max(1.20, vm.marThreshold + 0.30),
                        warnBelow: false,
                        onDecrease: { vm.adjustMARThreshold(delta: -0.05) },
                        onIncrease: { vm.adjustMARThreshold(delta: +0.05) }
                    )

                    // Head Pose Angles (Displays Relative Deviation from Road Baseline)
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Road Yaw: \(String(format: "%+.1f°", vm.deltaYaw))")
                                .foregroundColor(abs(vm.deltaYaw) > vm.predictor.yawThreshold ? .red : .green)
                            Text("Base: \(String(format: "%+.1f°", vm.baselineYaw))")
                                .font(.system(size: 10))
                                .foregroundColor(.gray)
                        }

                        Spacer()

                        VStack(alignment: .trailing, spacing: 2) {
                            Text("Road Pitch: \(String(format: "%+.1f°", vm.deltaPitch))")
                                .foregroundColor(vm.deltaPitch > vm.predictor.pitchUpThreshold || vm.deltaPitch < -vm.predictor.pitchDownThreshold ? .red : .green)
                            Text("Base: \(String(format: "%+.1f°", vm.baselinePitch))")
                                .font(.system(size: 10))
                                .foregroundColor(.gray)
                        }
                    }
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                }
                .padding(14)
                .background(Color.black.opacity(0.75))
                .cornerRadius(18)
                .padding(.horizontal)
                .padding(.bottom, 20)
            }
        }
        .onAppear {
            vm.start()
            flashAlert = true
        }
        .onDisappear {
            vm.stop()
        }
    }

    private func stateColor(_ state: String) -> Color {
        switch state {
        case "Focused": return .green
        case "Checking Mirror": return .cyan
        case "Drowsy": return .red
        case "Yawning": return .yellow
        case "Looking Away": return .orange
        default: return .gray
        }
    }
}

public struct TelemetryBar: View {
    public let title: String
    public let value: Float
    public let threshold: Float
    public let maxValue: Float
    public let warnBelow: Bool
    public var onDecrease: (() -> Void)? = nil
    public var onIncrease: (() -> Void)? = nil

    public var body: some View {
        VStack(spacing: 3) {
            HStack {
                Text(title)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white)
                Spacer()

                // Nudge buttons
                if let dec = onDecrease, let inc = onIncrease {
                    HStack(spacing: 6) {
                        Button(action: dec) {
                            Text("-")
                                .font(.system(size: 11, weight: .bold))
                                .frame(width: 18, height: 16)
                                .background(Color.white.opacity(0.2))
                                .cornerRadius(4)
                        }
                        Button(action: inc) {
                            Text("+")
                                .font(.system(size: 11, weight: .bold))
                                .frame(width: 18, height: 16)
                                .background(Color.white.opacity(0.2))
                                .cornerRadius(4)
                        }
                    }
                    .foregroundColor(.white)
                    .padding(.trailing, 4)
                }

                Text(String(format: "%.2f (thresh: %.2f)", value, threshold))
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundColor(isWarning ? .red : .green)
            }

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    // Track
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.white.opacity(0.2))

                    // Fill bar
                    RoundedRectangle(cornerRadius: 3)
                        .fill(isWarning ? Color.red : Color.green)
                        .frame(width: geo.size.width * CGFloat(min(1.0, max(0.0, value / maxValue))))

                    // Threshold Line
                    Rectangle()
                        .fill(Color.yellow)
                        .frame(width: 2, height: geo.size.height + 4)
                        .offset(x: geo.size.width * CGFloat(min(1.0, max(0.0, threshold / maxValue))) - 1, y: -2)
                }
            }
            .frame(height: 7)
        }
    }

    private var isWarning: Bool {
        warnBelow ? (value < threshold) : (value > threshold)
    }
}
