//
//  DMSPredictor.swift
//  Driver Monitoring System
//
//  Predicts driver state (Focused, Drowsy, Yawning, Looking Away) using EAR, MAR, and head pose.
//

import Foundation
import CoreML

public struct PredictionResult: Sendable {
    public let state: String
    public let confidence: Float
    public let ear: Float
    public let mar: Float
    public let yaw: Float
    public let pitch: Float
    public let deltaYaw: Float
    public let deltaPitch: Float
    public let isAlert: Bool
    public let lookawaySeconds: Float

    public init(
        state: String,
        confidence: Float,
        ear: Float,
        mar: Float,
        yaw: Float,
        pitch: Float,
        deltaYaw: Float,
        deltaPitch: Float,
        isAlert: Bool,
        lookawaySeconds: Float = 0.0
    ) {
        self.state = state
        self.confidence = confidence
        self.ear = ear
        self.mar = mar
        self.yaw = yaw
        self.pitch = pitch
        self.deltaYaw = deltaYaw
        self.deltaPitch = deltaPitch
        self.isAlert = isAlert
        self.lookawaySeconds = lookawaySeconds
    }
}

public final class DMSPredictor: @unchecked Sendable {
    public static let classes = ["Focused", "Drowsy", "Yawning", "Looking Away"]

    // Idle baselines measured when looking forward at the road
    public var baselineYaw: Float = 0.0
    public var baselinePitch: Float = 0.0
    public var idleMAR: Float = 0.35
    public var idleEAR: Float = 0.26

    // Adaptive thresholds
    public var earThreshold: Float = 0.15     // Default safe lower threshold
    public var marThreshold: Float = 0.75     // Default high yawn threshold (prevents false yawns at 45°)
    public var marDeltaBuffer: Float = 0.26   // Requires mouth to open +0.26 ABOVE resting state
    public var earScaleRatio: Float = 0.58    // Requires eye to drop below 58% of resting opening

    // Generous head pose tolerance (Euro NCAP / SAE Automotive standard)
    public var yawThreshold: Float = 28.0       // Allows wide scanning before flagging distraction
    public var pitchUpThreshold: Float = 24.0   // Upward tilt
    public var pitchDownThreshold: Float = 22.0 // Downward tilt (allows checking speedometer/gauges)

    public var pitchThreshold: Float {
        get { pitchUpThreshold }
        set {
            pitchUpThreshold = newValue
            pitchDownThreshold = newValue * 0.85
        }
    }

    // Consecutive frames tracking (@ 30 FPS)
    private var drowsyFrames = 0
    private var yawnFrames = 0
    private var lookawayFrames = 0

    private let drowsyThresholdFrames = 18 // ~0.6 - 1.0 sec (fast emergency alert for closed eyes)
    private let yawnThresholdFrames = 25   // ~0.8 - 1.2 sec sustained yawn
    // 75 frames @ 30 FPS = 2.5 SECONDS! Gives generous leeway for mirror & blind-spot checks!
    private let poseThresholdFrames = 75

    public init() {}

    /// Apply dynamic calibration based on measured resting driving posture
    public func applyCalibration(avgEAR: Float, avgMAR: Float, avgYaw: Float, avgPitch: Float) {
        self.baselineYaw = avgYaw
        self.baselinePitch = avgPitch
        self.idleEAR = avgEAR
        self.idleMAR = avgMAR

        // Dynamic Eye Threshold: 58% of resting eye opening
        self.earThreshold = max(0.11, min(0.20, avgEAR * earScaleRatio))

        // Dynamic Yawn Threshold: Idle MAR + 0.26 buffer
        self.marThreshold = max(0.68, avgMAR + marDeltaBuffer)
    }

    /// Classify driver state using adaptive thresholding relative to calibrated baseline
    public func predict(
        ear: Float,
        mar: Float,
        yaw: Float,
        pitch: Float,
        roll: Float,
        mouthOpeningRatio: Float = 0.0
    ) -> PredictionResult {
        // 1. Angular deviation from driver's calibrated forward road orientation
        let deltaYaw = yaw - baselineYaw
        let deltaPitch = pitch - baselinePitch

        var rawState = "Focused"
        var confidence: Float = 0.95

        // PRIORITY 1: Head Distraction (Only flags if head turns > 28° or tilts down > 22°)
        if abs(deltaYaw) > yawThreshold || deltaPitch > pitchUpThreshold || deltaPitch < -pitchDownThreshold {
            rawState = "Looking Away"
            confidence = min(1.0, max(abs(deltaYaw) / 45.0, abs(deltaPitch) / 30.0))
        }
        // PRIORITY 2: Yawning (Exceeds dynamic threshold + physical vertical aperture)
        else if mar > marThreshold && (mouthOpeningRatio > 0.08 || mouthOpeningRatio == 0.0) {
            rawState = "Yawning"
            confidence = min(1.0, max(0.65, (mar - idleMAR) / marDeltaBuffer))
        }
        // PRIORITY 3: Drowsiness / Microsleep (Eyes genuinely shut while facing the road)
        else if ear < earThreshold {
            rawState = "Drowsy"
            confidence = min(1.0, max(0.65, (earThreshold - ear) / earThreshold + 0.6))
        }

        // --- TEMPORAL PERSISTENCE FILTER ---
        var filteredState = "Focused"
        var isAlert = false

        if rawState == "Looking Away" {
            lookawayFrames += 1
            // While checking mirrors, suppress drowsy and yawn counters
            drowsyFrames = max(0, drowsyFrames - 2)
            yawnFrames = max(0, yawnFrames - 2)
        } else if rawState == "Drowsy" {
            drowsyFrames += 1
            yawnFrames = max(0, yawnFrames - 1)
            lookawayFrames = max(0, lookawayFrames - 4) // Fast reset when back facing road
        } else if rawState == "Yawning" {
            yawnFrames += 1
            drowsyFrames = max(0, drowsyFrames - 1)
            lookawayFrames = max(0, lookawayFrames - 4)
        } else {
            // Looking at the road: decay all alert counters
            drowsyFrames = max(0, drowsyFrames - 2)
            yawnFrames = max(0, yawnFrames - 2)
            lookawayFrames = max(0, lookawayFrames - 4)
        }

        let lookawaySec = Float(lookawayFrames) / 30.0

        // Resolve active alert state:
        // Distraction alarm triggers ONLY after 2.5 seconds (75 frames) of sustained off-road gaze!
        if lookawayFrames >= poseThresholdFrames {
            filteredState = "Looking Away"
            isAlert = true
        } else if drowsyFrames >= drowsyThresholdFrames {
            filteredState = "Drowsy"
            isAlert = true
        } else if yawnFrames >= yawnThresholdFrames {
            filteredState = "Yawning"
            isAlert = true
        } else if lookawayFrames > 15 {
            // Between 0.5s and 2.5s: Driver is checking mirrors/speedometer
            // State is shown on screen without sounding the acoustic alarm!
            filteredState = "Checking Mirror"
            isAlert = false
        }

        return PredictionResult(
            state: filteredState,
            confidence: confidence,
            ear: ear,
            mar: mar,
            yaw: yaw,
            pitch: pitch,
            deltaYaw: deltaYaw,
            deltaPitch: deltaPitch,
            isAlert: isAlert,
            lookawaySeconds: lookawaySec
        )
    }

    public func resetCounters() {
        drowsyFrames = 0
        yawnFrames = 0
        lookawayFrames = 0
    }
}
