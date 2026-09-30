//
//  AlertService.swift
//  Driver Monitoring System
//
//  Plays alert sounds and vibration feedback when the driver is drowsy or distracted.
//

import Foundation
import AudioToolbox
import UIKit

public final class AlertService: @unchecked Sendable {
    public static let shared = AlertService()

    private var lastAlertTime: TimeInterval = 0.0
    private let alertCooldown: TimeInterval = 1.0 // 1.0 second cooldown

    private init() {}

    /// Triggers acoustic warning and haptic vibration for the given driver state.
    public func triggerAlert(state: String) {
        let now = Date().timeIntervalSince1970
        guard now - lastAlertTime >= alertCooldown else { return }
        lastAlertTime = now

        DispatchQueue.main.async {
            let generator = UINotificationFeedbackGenerator()
            generator.prepare()

            if state == "Drowsy" {
                // High urgency alarm sound (System Sound 1005: SMS Alert or 1052)
                AudioServicesPlaySystemSound(1005)
                generator.notificationOccurred(.error)
            } else if state == "Yawning" {
                // Gentle chime (System Sound 1003)
                AudioServicesPlaySystemSound(1003)
                generator.notificationOccurred(.warning)
            } else if state == "Looking Away" {
                // Distinct double-tone alert
                AudioServicesPlaySystemSound(1052)
                generator.notificationOccurred(.warning)
            }
        }
    }
}
