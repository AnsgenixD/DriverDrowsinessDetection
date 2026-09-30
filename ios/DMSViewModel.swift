//
//  DMSViewModel.swift
//  Driver Monitoring System
//
//  Connects the camera feed and face landmark detection with the UI.
//

import Foundation
import SwiftUI
import AVFoundation
import Vision

public final class DMSViewModel: ObservableObject, CameraManagerDelegate, @unchecked Sendable {
    @Published public var state: String = "Focused"
    @Published public var confidence: Float = 0.95
    @Published public var ear: Float = 0.26
    @Published public var mar: Float = 0.35
    @Published public var yaw: Float = 0.0
    @Published public var pitch: Float = 0.0
    @Published public var deltaYaw: Float = 0.0
    @Published public var deltaPitch: Float = 0.0
    @Published public var fps: Double = 0.0
    @Published public var isAlertActive: Bool = false
    @Published public var faceDetected: Bool = false
    @Published public var isCalibrating: Bool = false
    @Published public var calibrationMessage: String = ""
    @Published public var earThreshold: Float = 0.15
    @Published public var marThreshold: Float = 0.75
    @Published public var baselineYaw: Float = 0.0
    @Published public var baselinePitch: Float = 0.0

    public let cameraManager = CameraManager()
    public let predictor = DMSPredictor()
    private let alertService = AlertService.shared

    private let sequenceHandler = VNSequenceRequestHandler()
    private var lastFrameTime = CFAbsoluteTimeGetCurrent()
    private var earSamples: [Float] = []
    private var marSamples: [Float] = []
    private var yawSamples: [Float] = []
    private var pitchSamples: [Float] = []

    public init() {
        cameraManager.delegate = self
        self.earThreshold = predictor.earThreshold
        self.marThreshold = predictor.marThreshold
    }

    public func start() {
        cameraManager.startSession()
    }

    public func stop() {
        cameraManager.stopSession()
    }

    /// Instant Zeroing: Sets current head position as the forward road heading
    public func setRoadPoseBaseline() {
        predictor.baselineYaw = yaw
        predictor.baselinePitch = pitch
        baselineYaw = yaw
        baselinePitch = pitch
        calibrationMessage = "Road Heading Set: Yaw \(Int(yaw))°"
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            self?.calibrationMessage = ""
        }
    }

    /// Complete 2-Second Adaptive Calibration:
    /// Measures driver's natural posture at their exact mount angle (e.g. 45° side mount)
    /// and establishes custom relative thresholds with large headroom.
    public func startCalibration() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.isCalibrating = true
            self.earSamples.removeAll()
            self.marSamples.removeAll()
            self.yawSamples.removeAll()
            self.pitchSamples.removeAll()
            self.calibrationMessage = "Hold driving posture & look at road..."
        }

        DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let self = self else { return }
            DispatchQueue.main.async {
                if !self.earSamples.isEmpty && !self.marSamples.isEmpty {
                    let avgEAR = self.earSamples.reduce(0, +) / Float(self.earSamples.count)
                    let avgMAR = self.marSamples.reduce(0, +) / Float(self.marSamples.count)
                    let avgYaw = self.yawSamples.reduce(0, +) / Float(self.yawSamples.count)
                    let avgPitch = self.pitchSamples.reduce(0, +) / Float(self.pitchSamples.count)

                    self.predictor.applyCalibration(
                        avgEAR: avgEAR,
                        avgMAR: avgMAR,
                        avgYaw: avgYaw,
                        avgPitch: avgPitch
                    )

                    self.earThreshold = self.predictor.earThreshold
                    self.marThreshold = self.predictor.marThreshold
                    self.baselineYaw = avgYaw
                    self.baselinePitch = avgPitch

                    self.calibrationMessage = "Calibrated! Idle MAR: \(String(format: "%.2f", avgMAR)) -> Yawn: \(String(format: "%.2f", self.marThreshold)) | Eyes: \(String(format: "%.2f", self.earThreshold))"
                } else {
                    self.calibrationMessage = "No face detected during calibration."
                }
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) { [weak self] in
                guard let self = self else { return }
                self.isCalibrating = false
                self.calibrationMessage = ""
            }
        }
    }

    /// Manual sensitivity nudges for fine-tuning
    public func adjustMARThreshold(delta: Float) {
        predictor.marThreshold = max(0.50, min(1.20, predictor.marThreshold + delta))
        marThreshold = predictor.marThreshold
    }

    public func adjustEARThreshold(delta: Float) {
        predictor.earThreshold = max(0.08, min(0.25, predictor.earThreshold + delta))
        earThreshold = predictor.earThreshold
    }

    // CameraManagerDelegate
    public func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer) {
        let currentTime = CFAbsoluteTimeGetCurrent()
        let dt = currentTime - lastFrameTime

        let bufferWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let bufferHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        let request = VNDetectFaceLandmarksRequest { [weak self] (req, err) in
            guard let self = self else { return }
            guard let results = req.results as? [VNFaceObservation], let face = results.first else {
                DispatchQueue.main.async {
                    self.faceDetected = false
                }
                return
            }

            self.processFaceObservation(face: face, dt: dt, imgW: bufferWidth, imgH: bufferHeight)
        }

        try? sequenceHandler.perform([request], on: pixelBuffer)
        lastFrameTime = currentTime
    }

    private func processFaceObservation(face: VNFaceObservation, dt: Double, imgW: CGFloat, imgH: CGFloat) {
        guard let landmarks = face.landmarks,
              let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let outerLips = landmarks.outerLips else {
            return
        }

        let bbox = face.boundingBox

        // 1. Pixel-Accurate EAR for left and right eyes
        let earL = calculateEyeEAR(region: leftEye, bbox: bbox, w: imgW, h: imgH)
        let earR = calculateEyeEAR(region: rightEye, bbox: bbox, w: imgW, h: imgH)
        let earAvg = (earL + earR) / 2.0

        // 2. Pixel-Accurate MAR and vertical opening ratio
        let (mar, openingRatio) = calculateLipMetrics(region: outerLips, bbox: bbox, w: imgW, h: imgH)

        // 3. Head Pose (Euler Angles)
        let yawDeg = Float(face.yaw?.doubleValue ?? 0.0) * (180.0 / .pi)
        let pitchDeg = Float(face.pitch?.doubleValue ?? 0.0) * (180.0 / .pi)
        let rollDeg = Float(face.roll?.doubleValue ?? 0.0) * (180.0 / .pi)

        // 4. Run Prediction with Dynamic Baseline Headroom
        let result = predictor.predict(
            ear: earAvg,
            mar: mar,
            yaw: yawDeg,
            pitch: pitchDeg,
            roll: rollDeg,
            mouthOpeningRatio: openingRatio
        )

        // 5. Trigger Sound/Haptics if alerting
        if result.isAlert {
            alertService.triggerAlert(state: result.state)
        }

        // Update UI state safely on main thread
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.faceDetected = true
            self.state = result.state
            self.confidence = result.confidence
            self.ear = result.ear
            self.mar = result.mar
            self.yaw = yawDeg
            self.pitch = pitchDeg
            self.deltaYaw = result.deltaYaw
            self.deltaPitch = result.deltaPitch
            self.isAlertActive = result.isAlert
            self.fps = dt > 0 ? (1.0 / dt) : 30.0

            if self.isCalibrating {
                self.earSamples.append(earAvg)
                self.marSamples.append(mar)
                self.yawSamples.append(yawDeg)
                self.pitchSamples.append(pitchDeg)
            }
        }
    }

    private func calculateEyeEAR(region: VNFaceLandmarkRegion2D, bbox: CGRect, w: CGFloat, h: CGFloat) -> Float {
        let rawPts = region.normalizedPoints
        guard rawPts.count >= 6 else { return 0.28 }
        let pts = rawPts.map { GeometryUtils.toPixel(pt: $0, bbox: bbox, imgW: w, imgH: h) }

        if pts.count == 8 {
            let p0 = pts[0]
            let p4 = pts[4]
            let p1 = pts[1]
            let p7 = pts[7]
            let p3 = pts[3]
            let p5 = pts[5]

            let horizontal = p0.distance(to: p4)
            if horizontal < 1e-4 { return 0.28 }

            let v1 = p1.distance(to: p7)
            let v2 = p3.distance(to: p5)

            // Equation 1: (||p1 - p7|| + ||p3 - p5||) / (2 * ||p0 - p4||)
            return Float((v1 + v2) / (2.0 * horizontal))
        } else {
            return GeometryUtils.calculateEAR(pts: Array(pts.prefix(6)))
        }
    }

    private func calculateLipMetrics(region: VNFaceLandmarkRegion2D, bbox: CGRect, w: CGFloat, h: CGFloat) -> (mar: Float, openingRatio: Float) {
        let rawPts = region.normalizedPoints
        guard rawPts.count >= 8 else { return (0.15, 0.0) }
        let pts = rawPts.map { GeometryUtils.toPixel(pt: $0, bbox: bbox, imgW: w, imgH: h) }
        let faceHeight = bbox.size.height * h

        if pts.count >= 14 {
            let qLeft = pts[0]
            let qRight = pts[7]
            let horizontal = qLeft.distance(to: qRight)
            if horizontal < 1e-4 { return (0.15, 0.0) }

            let v1 = pts[2].distance(to: pts[12])  // vertical left pair
            let v2 = pts[4].distance(to: pts[10])  // vertical center pair
            let v3 = pts[6].distance(to: pts[8])   // vertical right pair

            let mar = Float((v1 + v2 + v3) / (3.0 * horizontal))
            let openingRatio = Float(v2 / max(1.0, faceHeight))

            return (mar, openingRatio)
        } else {
            let xs = pts.map { $0.x }
            let ys = pts.map { $0.y }
            let mouthW = max(1.0, (xs.max() ?? 1) - (xs.min() ?? 0))
            let mouthH = (ys.max() ?? 1) - (ys.min() ?? 0)
            let mar = Float(mouthH / mouthW)
            let openingRatio = Float(mouthH / max(1.0, faceHeight))
            return (mar, openingRatio)
        }
    }
}
