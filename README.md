# YOLO11 + Kalman Filter + SIFT Feature Matching Object Tracking System

This is an advanced object tracking system based on YOLO11, Kalman filtering, and SIFT feature matching, with visual speed prediction, motion prediction, depth estimation, occlusion handling, trajectory drawing, hover information display, and complete configuration management.

## Main Features

### 🎯 Core Features
- **YOLO11 Object Detection**: Real-time object detection using the latest YOLO11 model
- **Kalman Filter Tracking**: Smooth object tracking and state prediction
- **SIFT Feature Matching**: Extract and match SIFT feature points for improved tracking accuracy
- **Multi-Object Tracking**: Track multiple objects simultaneously with automatic ID assignment
- **Occlusion Handling**: Continue predicting positions when objects are occluded without losing tracking
- **Speed Prediction**: Real-time calculation and display of object movement speed
- **Depth Estimation**: Relative depth distance estimation based on object size
- **Trajectory Drawing**: Display complete movement trajectory within 3 seconds
- **Future Prediction**: Predict future positions based on current velocity
- **Hover Information Display**: Show detailed information when hovering over confidence values

### 🎨 Visualization Features
- **Dual Display**: Simultaneously show detection boxes (green) and prediction boxes (colored)
- **Enhanced Detection Boxes**: Green boxes with corner markers for YOLO detection results
- **Enhanced Tracking Boxes**: Colored boxes with depth-coded backgrounds for Kalman tracking
- **Depth Markers**: Color-coded cross marks and corner dots indicating object depth
- **Status Indication**: Solid lines for normal tracking, dashed lines for occlusion prediction
- **Detailed Info Boxes**: Display confidence, speed, direction, depth and other detailed information
- **Velocity Vectors**: Arrows showing object movement direction and speed magnitude
- **Direction Indication**: Use arrow symbols and angles to show movement direction
- **Trajectory Lines**: Gradient colors showing historical movement paths
- **Feature Points**: Display SIFT feature point locations
- **Information Display**: Real-time FPS, detection count, tracking count, etc.
- **Interactive Hover**: Mouse hover over confidence values to see detailed tracking information
- **Depth Color Legend**: Visual reference for depth color coding

### 🆕 New Features
- **SIFT Feature Matching**: Improve target association accuracy and reduce ID switching
- **Depth Estimation**: Calculate relative depth distance based on object size
- **Enhanced Depth Visualization**: Color-coded depth markers with cross marks and corner dots
- **Dual Box Display**: Simultaneous display of detection boxes (green) and tracking boxes (colored)
- **Enhanced Box Styling**: Corner markers for detection boxes, depth-colored backgrounds for tracking boxes
- **Motion Analysis**: Detailed speed and direction statistics
- **Smart Information Display**: Adaptive positioning of detailed information boxes
- **Feature Visualization**: Display key feature point locations
- **All COCO Classes Detection**: Detect all 80 object classes (person, car, bicycle, animals, furniture, etc.)
- **Improved Depth Color Coding**: Six-level depth classification with intuitive color mapping
- **Configuration Management System**: Complete configuration management and persistence
- **Multi-Window Monitoring**: Support multiple monitoring windows for simultaneous display
- **Real-time UI Control**: Dynamically toggle display elements during runtime
- **Mouse Interaction**: Hover over UI elements to see detailed information

### ⚙️ Configuration Management System
- **Auto Save**: Automatically save configuration to config.json when program exits
- **Real-time Adjustment**: Use keyboard shortcuts to adjust display options during runtime
- **GUI Configuration Tool**: Graphical configuration management interface
- **Camera Detection**: Automatically detect available camera devices
- **Multi-Window Configuration**: Flexible configuration of multiple monitoring windows
- **Parameter Tuning**: Detailed tracking and performance parameter settings

## System Requirements

### Hardware Requirements
- **CPU**: AMD Ryzen 5 5600 or equivalent performance
- **GPU**: NVIDIA GeForce RTX 2060 (CUDA support)
- **Memory**: At least 8GB RAM
- **Storage**: At least 2GB available space

### Software Requirements
- **Operating System**: Windows 10 22H2 64-bit
- **Python**: 3.8+
- **CUDA**: Supported NVIDIA drivers
- **OpenCV**: Including contrib module (SIFT support)

## Installation Instructions

### 1. Clone Project
```bash
git clone <repository-url>
cd yolo-kalman-tracker
```

### 2. Run Installation Script
```bash
# Windows
install.bat

# Or manual installation
pip install -r requirements.txt
```

### 3. Download YOLO Model
The program will automatically download the YOLO11 model, or you can download manually:
```bash
# Auto download (on first run)
python demo.py

# Or manually download specific model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
```

## Usage

### Basic Usage

#### 1. Use Camera (Recommended)
```bash
python demo.py
```

#### 2. Specify Camera ID
```bash
python demo.py --camera 0    # Use camera 0
python demo.py --camera 1    # Use camera 1
```

#### 3. Process Video File
```bash
python demo.py --source video.mp4
```

#### 4. Save Output Video
```bash
python demo.py --source video.mp4 --output output.mp4
```

### Configuration Management

#### 1. List Available Cameras
```bash
python demo.py --list-cameras
```

#### 2. View Current Configuration
```bash
python demo.py --config
```

#### 3. Reset Configuration
```bash
python demo.py --reset-config
```

#### 4. Use GUI Configuration Tool
```bash
python config_gui.py
```

#### 5. Test Hover Functionality
```bash
python test_hover.py
```

### Advanced Parameters

```bash
python demo.py \
    --source 0 \                    # Video source (number=camera ID, or file path)
    --camera 1 \                    # Specify camera ID
    --output output.mp4 \           # Output video path
    --model yolo11s.pt \           # YOLO model (n/s/m/l/x)
    --conf 0.5 \                   # Detection confidence threshold
    --no-ui                        # Disable UI display
```

### Real-time Control

The following keyboard shortcuts are available during program execution:

- **1**: Toggle detection boxes display
- **2**: Toggle tracking boxes display
- **3**: Toggle trajectory display
- **4**: Toggle velocity vectors display
- **5**: Toggle future predictions display
- **6**: Toggle SIFT features display
- **7**: Toggle detailed info display
- **8**: Toggle FPS info display
- **9**: Toggle legend display
- **0**: Toggle depth markers display
- **-**: Toggle depth legend display
- **r**: Reset configuration to default values
- **s**: Save current configuration
- **h**: Show help information
- **q**: Exit program

### Mouse Interaction

- **Hover over confidence values**: See detailed tracking information in tooltip
- **Move mouse over info boxes**: Display comprehensive object details including:
  - Object ID and tracking status
  - Detection confidence
  - Movement speed (pixels/frame)
  - Direction (arrow symbol and angle)
  - Estimated depth distance with color coding
  - Velocity components

### Depth Color Coding

The system uses a six-level depth classification with intuitive color mapping:

- **Red**: Very Close (<200px) - Objects very close to camera
- **Orange**: Close (200-500px) - Objects in close range
- **Yellow**: Near (500-1000px) - Objects in near range
- **Green**: Medium (1000-2000px) - Objects at medium distance
- **Cyan**: Far (2000-3000px) - Objects at far distance
- **Blue**: Very Far (>3000px) - Objects very far from camera

Depth markers include:
- **Cross marks**: Central depth indicators on each tracked object
- **Corner dots**: Depth-colored markers at object corners
- **Background colors**: Tracking box backgrounds use depth colors with transparency

### Available YOLO Models
- `yolo11n.pt`: Fastest, lower accuracy
- `yolo11s.pt`: Balanced speed and accuracy (recommended)
- `yolo11m.pt`: Medium model
- `yolo11l.pt`: Large model
- `yolo11x.pt`: Largest model, highest accuracy

## Program Architecture

### Core Modules

#### 1. `config.py`
- **Config Class**: Configuration management core class
- **Auto Save**: Automatically save configuration when program exits
- **Deep Merge**: Intelligently merge default and user configurations
- **Camera Detection**: Automatically detect available camera devices
- **Configuration Validation**: Validate configuration parameter validity

#### 2. `kalman_tracker.py`
- **KalmanTracker Class**: Single object Kalman filter tracker
- **State Management**: 8-dimensional state vector for position, velocity, and size
- **Feature Extraction**: SIFT feature point detection and descriptor computation
- **Depth Estimation**: Depth calculation based on object size
- **Motion Analysis**: Speed and direction statistics

#### 3. `multi_object_tracker.py`
- **MultiObjectTracker Class**: Multi-object tracking manager
- **Data Association**: Matching algorithm combining IoU, distance, and SIFT features
- **Feature Matching**: SIFT descriptor matching and similarity computation
- **Lifecycle Management**: Automatically create and delete trackers

#### 4. `yolo_tracker.py`
- **YOLOTracker Class**: Main tracking system
- **Detection Integration**: Integration of YOLO11 detection with Kalman tracking
- **Visualization**: All drawing and display functions
- **Information Display**: Detailed information boxes and legends
- **Configuration Integration**: Complete configuration system support
- **Multi-Window Management**: Support for multiple monitoring windows
- **Mouse Interaction**: Hover functionality for detailed information display

#### 5. `demo.py`
- **Command Line Interface**: Provides easy-to-use command line parameters
- **Configuration Management**: Handle various input/output options
- **Camera Detection**: Automatically detect and configure cameras

#### 6. `config_gui.py`
- **ConfigGUI Class**: Graphical configuration management tool
- **Tabbed Interface**: Categorized management of different configuration types
- **Real-time Preview**: Real-time preview of configuration changes
- **File Management**: Configuration file loading and saving

#### 7. `test_hover.py`
- **Test Script**: Simple test script for hover functionality
- **Interactive Testing**: Test mouse hover information display
- **Feature Validation**: Validate new interactive features

## Configuration System Details

### Configuration File Structure (config.json)

```json
{
    "camera_id": 0,
    "model_path": "yolo11n.pt",
    "confidence_threshold": 0.5,
    "camera_settings": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "auto_exposure": true,
        "brightness": 0,
        "contrast": 0,
        "saturation": 0
    },
    "show_ui": true,
    "show_detection_boxes": true,
    "show_tracking_boxes": true,
    "show_trajectories": true,
    "show_velocity_vectors": true,
    "show_future_predictions": true,
    "show_sift_features": true,
    "show_detailed_info": true,
    "show_fps_info": true,
    "show_legend": true,
    "monitor_windows": {
        "main_window": {
            "enabled": true,
            "name": "YOLO11 + Kalman + SIFT Tracking",
            "position": [100, 100],
            "size": [1280, 720]
        },
        "detection_window": {
            "enabled": false,
            "name": "Detection Only",
            "position": [1400, 100],
            "size": [640, 480]
        },
        "tracking_window": {
            "enabled": false,
            "name": "Tracking Only",
            "position": [1400, 600],
            "size": [640, 480]
        }
    },
    "tracking_params": {
        "max_disappeared": 30,
        "max_distance": 100,
        "trajectory_length": 90,
        "future_prediction_steps": 15,
        "future_prediction_display_steps": 3,
        "sift_feature_limit": 10
    },
    "display_params": {
        "font_scale": 0.7,
        "line_thickness": 2,
        "box_thickness": 2,
        "text_color": [255, 255, 255],
        "background_alpha": 0.7
    },
    "output_settings": {
        "save_video": false,
        "output_path": "output.mp4",
        "fps": 30,
        "codec": "mp4v"
    },
    "performance": {
        "use_gpu": true,
        "max_fps": 30,
        "skip_frames": 0,
        "resize_input": false,
        "input_size": [640, 640]
    }
}
```

### Configuration Parameters Description

#### Basic Settings
- `camera_id`: Camera device ID (0, 1, 2...)
- `model_path`: YOLO model file path
- `confidence_threshold`: Detection confidence threshold (0.1-1.0)

#### Camera Settings
- `width`: Camera resolution width (default: 1920)
- `height`: Camera resolution height (default: 1080)
- `fps`: Camera frame rate (default: 30)
- `auto_exposure`: Enable automatic exposure
- `brightness`: Brightness adjustment (-100 to 100)
- `contrast`: Contrast adjustment (-100 to 100)
- `saturation`: Saturation adjustment (-100 to 100)

#### UI Display Control
- `show_ui`: Master UI switch
- `show_detection_boxes`: Display YOLO detection boxes
- `show_tracking_boxes`: Display Kalman tracking boxes
- `show_trajectories`: Display movement trajectories
- `show_velocity_vectors`: Display velocity vectors
- `show_future_predictions`: Display future position predictions
- `show_sift_features`: Display SIFT feature points
- `show_detailed_info`: Display detailed information boxes
- `show_fps_info`: Display FPS and statistics
- `show_legend`: Display legend

#### Window Configuration
- `enabled`: Whether window is enabled
- `name`: Window title
- `position`: Window position [x, y]
- `size`: Window size [width, height]

#### Tracking Parameters
- `max_disappeared`: Maximum frames for object disappearance
- `max_distance`: Maximum association distance
- `trajectory_length`: Trajectory history length (frames)
- `future_prediction_steps`: Total future prediction steps for internal calculation
- `future_prediction_display_steps`: Number of future prediction points to display (default: 3)
- `sift_feature_limit`: SIFT feature point display limit

#### Display Parameters
- `font_scale`: Text font scale
- `line_thickness`: Line thickness for drawing
- `box_thickness`: Box border thickness
- `text_color`: Text color [R, G, B]
- `background_alpha`: Background transparency

#### Output Settings
- `save_video`: Whether to save output video
- `output_path`: Output video file path
- `fps`: Output video frame rate
- `codec`: Video codec

#### Performance Settings
- `use_gpu`: Enable GPU acceleration
- `max_fps`: Maximum frame rate limit
- `skip_frames`: Number of frames to skip
- `resize_input`: Whether to resize input
- `input_size`: Input image size [width, height]

## Technical Details

### Kalman Filter Configuration
```python
# 8-dimensional state vector: [x, y, w, h, vx, vy, vw, vh]
# x, y: center coordinates
# w, h: width and height
# vx, vy: velocity in x and y directions
# vw, vh: width and height change rates
```

### SIFT Feature Matching
- **Feature Detection**: Using OpenCV SIFT detector
- **Descriptor Matching**: BFMatcher with Lowe's ratio test
- **Matching Threshold**: 0.75 distance ratio
- **Cost Calculation**: 70% basic cost + 30% feature cost

### Depth Estimation Algorithm
```python
# Depth estimation based on object size
depth = reference_size * 1000 / sqrt(area)
# Range limit: 50-5000 pixels
```

### Data Association Algorithm
- **Cost Calculation**: IoU + Euclidean distance + SIFT feature similarity
- **Matching Algorithm**: Hungarian algorithm
- **Threshold Filtering**: Filter low-quality matches

### Occlusion Handling Strategy
- **Prediction Continuation**: Continue Kalman prediction when detection is lost
- **Feature Memory**: Save SIFT features for re-association
- **Survival Time**: Delete after maximum 30 frames (1 second) without detection
- **Visual Indication**: Dashed boxes indicate prediction state

## Display Information Description

### Detection Box Information
- **Green Box**: YOLO detection results
- **Label**: Class name and confidence

### Tracking Box Information
- **Colored Box**: Kalman filter tracking results
- **ID**: Unique tracking identifier
- **Status**: "Track" or "Lost"
- **Confidence**: Detection confidence (0.00-1.00)
- **Speed**: Pixels/frame (px/f)
- **Direction**: Arrow symbol and angle (°)
- **Depth**: Estimated depth distance (px)

### Visual Elements
- **Solid Box**: Normal tracking state
- **Dashed Box**: Occlusion prediction state
- **Arrow**: Velocity vector and direction
- **Trajectory Line**: 3-second movement history
- **Small Dots**: Future position predictions
- **Feature Points**: SIFT keypoints

### Interactive Elements
- **Hover Information**: Detailed tooltip when hovering over confidence values
- **Mouse Tracking**: Real-time mouse position tracking for interaction
- **Dynamic Display**: Information boxes that follow mouse movement
- **Comprehensive Details**: Full object information in hover tooltips

## Performance Optimization

### Recommended Settings
- **Detection Threshold**: 0.5 (balance accuracy and speed)
- **Model Selection**: yolo11s.pt (recommended)
- **Input Resolution**: 640x640 (YOLO default)
- **Feature Point Count**: Limit display to top 10

### Performance Monitoring
- **FPS Display**: Real-time frame rate monitoring
- **Object Count**: Detection and tracking quantity statistics
- **Memory Usage**: Trajectory history limited to 90 frames
- **Feature Matching**: Enable status display

### Configuration Optimization
- **Disable unnecessary UI elements**: Improve performance
- **Adjust tracking parameters**: Optimize based on scene
- **GPU acceleration**: Enable CUDA support
- **Frame skipping**: Reduce computational load

## Troubleshooting

### Common Issues

#### 1. SIFT Features Not Available
```bash
# Install opencv-contrib-python
pip install opencv-contrib-python>=4.8.0
```

#### 2. CUDA Errors
```bash
# Check CUDA installation
nvidia-smi
# Install CPU version PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Camera Cannot Open
```bash
# Check camera devices
python demo.py --list-cameras
```

#### 4. Model Download Failed
```bash
# Manually download model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt
```

#### 5. Configuration File Corrupted
```bash
# Reset configuration
python demo.py --reset-config
```

#### 6. Hover Information Not Working
```bash
# Test hover functionality
python test_hover.py
```

### Performance Tuning
- **Lower Resolution**: Use smaller input images
- **Adjust Detection Frequency**: Detect every N frames
- **Limit Tracking Count**: Set maximum tracking targets
- **Feature Point Limit**: Reduce SIFT feature point count
- **Disable UI Elements**: Turn off unnecessary display functions

## Extended Features

### Custom Detection Categories
```python
# Modify detection function in yolo_tracker.py
def detect_objects(self, frame):
    results = self.model(frame, classes=[0, 1, 2])  # Only detect specific classes
```

### Adjust Depth Estimation Parameters
```python
# Adjust depth calculation in multi_object_tracker.py
def estimate_depth_from_size(self, bbox, reference_size=100):
    # Adjust reference_size for different scenes
```

### Custom Feature Matching Threshold
```python
# Adjust matching parameters in multi_object_tracker.py
def match_features(self, desc1, desc2):
    # Adjust 0.75 threshold to change matching strictness
    if m.distance < 0.75 * n.distance:
```

### Add New UI Elements
```python
# Add new configuration item in config.py
'show_new_feature': True

# Add corresponding drawing function in yolo_tracker.py
def draw_new_feature(self, frame, data):
    if not self.config.get('show_new_feature', True):
        return
    # Drawing logic
```

### Enhance Mouse Interaction
```python
# Add more mouse events in yolo_tracker.py
def mouse_callback(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Handle left click
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Handle right click
```

## License and Contribution

This project is based on open source license, contributions and improvement suggestions are welcome.

### Contribution Guidelines
1. Fork the project
2. Create feature branch
3. Submit changes
4. Initiate Pull Request

## Contact Information

For questions or suggestions, please contact through:
- Submit Issue
- Send Pull Request
- Technical discussion

---

**Note**: 
- The YOLO model will be automatically downloaded on first run, please ensure network connection is normal
- SIFT features require opencv-contrib-python support
- GPU acceleration is recommended for best performance
- Configuration files are automatically saved when the program exits
- Use GUI configuration tool for more convenient settings management
- Hover over confidence values to see detailed tracking information
- Test hover functionality with test_hover.py script 

---

## 🤖 Development Partnership

This project was developed with assistance from **Claude.ai**, an AI assistant by Anthropic. Claude.ai provided support in the development process through:

- **Code Review**: Assistance in identifying potential issues and suggesting improvements
- **Problem Analysis**: Help with debugging and troubleshooting technical challenges
- **Documentation Support**: Assistance in creating comprehensive documentation
- **Feature Suggestions**: Providing ideas and recommendations for enhanced functionality
- **Best Practices**: Guidance on coding standards and optimization techniques

### Development Highlights
- **Human-Led Development**: Primary development driven by human creativity and expertise
- **AI-Assisted Review**: Claude.ai helped identify potential issues and suggest improvements
- **Collaborative Problem Solving**: AI assistance in analyzing complex technical challenges
- **Enhanced Documentation**: AI support in creating detailed guides and explanations
- **Quality Assurance**: Additional perspective on code quality and user experience

Claude.ai served as a helpful assistant, providing suggestions, identifying potential problems, and offering ideas to enhance the development process while the core development remained human-driven.

---

**Developed by Human Expertise with AI Assistance**

---

###### README documentation written by Claude.ai 