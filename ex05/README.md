# Browser-based Inferencing on Human centric models

A real-time computer vision system that combines hand gesture recognition and facial expression detection 
using TensorFlow.js and face-api.js.

### Problem Design

The project addresses the need for natural, non-verbal human-computer interaction through:

1. Real-time Detection Requirements:
   - Hand gesture recognition (thumbs up/down, raised hand)
   - Facial expression analysis
   - User presence monitoring
   - Maximum 500ms latency for gesture detection
   - Maximum 100ms latency for expression detection

2. System Constraints:
   - Browser-based implementation
   - Webcam access required
   - Modern browser with WebGL support
   - Sufficient CPU/GPU resources for real-time processing

3. Success Criteria:
   - Accurate gesture recognition in varying lighting conditions
   - Responsive expression detection with minimal latency
   - Clear visual and audio feedback
   - Graceful handling of detection failures

### Data Preparation

1. Video Stream Processing:
   - Real-time webcam feed capture
   - Frame buffering and synchronization
   - Resolution standardization (720x560)
   - Canvas overlay for visualization

2. Model Requirements:
   - Pre-trained TensorFlow.js models for hand detection
   - face-api.js models for expression recognition
   - Model files organization:
     ```
     /models
     ├── face_landmark_68.weights
     ├── face_expression.weights
     ├── face_recognition.weights
     └── tiny_face_detector.weights
     ```

### Code Organization

1. Project Structure:

```
├── index.html
├── lib
│   ├── blazeface.js
│   ├── face-api.min.js
│   ├── handpose.min.js
│   ├── mobilenet.js
│   ├── tfjs.js
│   └── tf.min.js
├── main.js
├── README.md
└── styles.css
```

2. Core Components:
   - Camera setup and video stream management
   - Model loading and initialization
   - Gesture detection pipeline
   - Expression recognition system
   - UI update and feedback mechanism

3. Key Functions:
   - `setupCamera()`: Video stream initialization
   - `loadModels()`: ML models loading
   - `detectGestures()`: Hand gesture processing
   - `setupExpressionDetection()`: Facial expression analysis
   - `updateExpressionUI()`: Interface updates

### Test Cases

1. Initialization Tests:
   - Camera access permission handling
   - Model loading verification
   - Audio preloading confirmation

2. Gesture Detection Tests:
   - Thumbs up recognition accuracy
   - Thumbs down detection reliability
   - Hand raised identification
   - Multiple gesture handling

3. Expression Recognition Tests:
   - Seven basic emotions detection
   - Expression transition smoothness
   - Multiple face handling
   - Lighting variation impact

4. System Reliability Tests:
   - Connection loss handling
   - CPU/Memory usage monitoring
   - Long-duration stability
   - Browser compatibility

### Further Optimizations and Improvements

1. Performance Enhancements:
   - WebGL acceleration optimization
   - Frame processing efficiency
   - Model loading time reduction
   - Memory management improvement

2. Feature Additions:
   - Custom gesture training
   - Expression intensity tracking
   - Multi-user support
   - Gesture sequence recognition

3. User Experience:
   - Customizable feedback options
   - Accessibility improvements
   - Mobile device optimization
   - Offline mode support
