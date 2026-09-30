//
//  GeometryUtils.swift
//  Driver Monitoring System
//
//  Helper functions to calculate Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
//

import Foundation
import CoreGraphics

public struct Point2D: Sendable {
    public var x: CGFloat
    public var y: CGFloat

    public init(x: CGFloat, y: CGFloat) {
        self.x = x
        self.y = y
    }

    public func distance(to other: Point2D) -> CGFloat {
        let dx = self.x - other.x
        let dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

public final class GeometryUtils: @unchecked Sendable {

    /// Convert Vision normalized landmark point to isotropic image pixel coordinates.
    /// Eliminates aspect ratio distortion between width and height.
    public static func toPixel(pt: CGPoint, bbox: CGRect, imgW: CGFloat, imgH: CGFloat) -> Point2D {
        let imgX = bbox.origin.x + pt.x * bbox.size.width
        let imgY = bbox.origin.y + pt.y * bbox.size.height
        return Point2D(x: imgX * imgW, y: imgY * imgH)
    }

    /// Calculate Eye Aspect Ratio (EAR) according to Equation 1:
    /// EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 * ||p1 - p4|| )
    public static func calculateEAR(pts: [Point2D]) -> Float {
        guard pts.count >= 6 else { return 0.28 }
        let p1 = pts[0]
        let p2 = pts[1]
        let p3 = pts[2]
        let p4 = pts[3]
        let p5 = pts[4]
        let p6 = pts[5]

        let vertical1 = p2.distance(to: p6)
        let vertical2 = p3.distance(to: p5)
        let horizontal = p1.distance(to: p4)

        if horizontal < 1e-4 { return 0.28 }
        return Float((vertical1 + vertical2) / (2.0 * horizontal))
    }

    /// Calculate Mouth Aspect Ratio (MAR) according to Equation 2:
    /// MAR = ( ||q2 - q8|| + ||q3 - q7|| + ||q4 - q6|| ) / ( 3 * ||q1 - q5|| )
    public static func calculateMAR(pts: [Point2D]) -> Float {
        guard pts.count >= 8 else { return 0.15 }
        let q1 = pts[0]
        let q2 = pts[1]
        let q3 = pts[2]
        let q4 = pts[3]
        let q5 = pts[4]
        let q6 = pts[5]
        let q7 = pts[6]
        let q8 = pts[7]

        let v1 = q2.distance(to: q8)
        let v2 = q3.distance(to: q7)
        let v3 = q4.distance(to: q6)
        let horizontal = q1.distance(to: q5)

        if horizontal < 1e-4 { return 0.15 }
        return Float((v1 + v2 + v3) / (3.0 * horizontal))
    }
}
