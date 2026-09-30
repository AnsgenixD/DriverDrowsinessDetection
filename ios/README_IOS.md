# iOS Deployment Guide — Driver Monitoring System (Swift & SwiftUI)

This directory contains the native Swift / SwiftUI implementation of the Driver Monitoring System (DMS).

---

## 1. Why iOS / iPhone is Ideal for this System

- **Apple Neural Engine (ANE)**: Runs CoreML models (`MobileNetV3_DMS.mlpackage`) at **60+ FPS** with negligible battery consumption.
- **AVFoundation**: Direct front-camera video streaming at 30/60 FPS with hardware mirroring.
- **Haptic Alerts (Taptic Engine)**: In addition to acoustic chimes, the iPhone physical vibration motor (`UINotificationFeedbackGenerator`) alerts the driver if they nod off.
- **Built-in Face Tracking**: Utilizes Apple's native Vision framework (`VNSequenceRequestHandler`) and MediaPipe for landmark tracking with zero battery lag.

---

## 2. Project File Structure

```
ios/
├── DMSApp.swift              # Main SwiftUI App entrypoint
├── ContentView.swift         # SwiftUI Heads-Up Display (HUD) with live gauges & warning banners
├── DMSViewModel.swift        # ObservableObject bridging camera, geometry, and UI state
├── DMSPredictor.swift        # CoreML & Geometric Rule Classifier (4-way state tracking)
├── CameraManager.swift       # AVFoundation 30 FPS front-camera capture pipeline
├── GeometryUtils.swift       # Swift calculations for EAR (Eq. 1), MAR (Eq. 2), and Head Pose
├── AlertService.swift        # Acoustic sound & Taptic Engine vibration engine
├── Info.plist                # Camera usage permission (NSCameraUsageDescription)
└── README_IOS.md             # This guide
```

---

## 3. How to Open and Run in Xcode (Step-by-Step)

### Step 1: Create or Open Project in Xcode
1. Open **Xcode** on your Mac.
2. Select **File -> New -> Project...**
3. Choose **iOS -> App**, click **Next**.
4. Set:
   - **Product Name**: `DriverMonitoringSystem`
   - **Interface**: `SwiftUI`
   - **Language**: `Swift`
5. Save the project to your chosen location.

### Step 2: Add the Swift Files
1. Drag the files from `ProjectExhibition1/ios/` into your Xcode project navigator:
   - `DMSApp.swift` (replace default)
   - `ContentView.swift` (replace default)
   - `DMSViewModel.swift`
   - `DMSPredictor.swift`
   - `CameraManager.swift`
   - `GeometryUtils.swift`
   - `AlertService.swift`
2. When prompted by Xcode, check **"Copy items if needed"** and ensure your app target is checked.

### Step 3: Add the CoreML Model
Drag either of your exported CoreML models from `weights/` into Xcode:
- `weights/MobileNetV3_DMS.mlpackage` (MobileNetV3 backbone)
- `weights/DriverStateClassifier.mlpackage` (Custom 2D CNN backbone)

Xcode will automatically generate a Swift interface for the model!

### Step 4: Camera Permission
In Xcode, click on your project's **Target -> Info**:
- Ensure `Privacy - Camera Usage Description` (`NSCameraUsageDescription`) is present:
  > *"Driver Monitoring System requires front-facing camera access to track eye closure, yawning, and head pose in real time."*

### Step 5: Run on iPhone
1. Connect your iPhone via USB / Wi-Fi.
2. Select your iPhone as the destination device in the Xcode toolbar.
3. Click the **Play / Run** button (`Cmd + R`).
4. Mount your phone on a car dashboard or laptop stand — live telemetry, EAR/MAR gauges, and acoustic/vibrational warnings will activate immediately!
