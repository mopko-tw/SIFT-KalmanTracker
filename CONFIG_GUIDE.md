# YOLO11 Object Tracking System Configuration Guide

## Configuration File: config.json

This document provides detailed explanations for all parameters in the `config.json` configuration file.

---

## 📋 Configuration Structure Overview

```json
{
    "camera_id": 0,
    "model_path": "yolo11n.pt",
    "confidence_threshold": 0.5,
    "camera_settings": { ... },
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
    "show_depth_markers": true,
    "show_depth_legend": true,
    "monitor_windows": { ... },
    "tracking_params": { ... },
    "display_params": { ... },
    "output_settings": { ... },
    "performance": { ... }
}
```

---

## 🎯 Basic Settings

### `camera_id`
- **Type**: Integer or String
- **Default**: `0`
- **Description**: Camera device ID or video file path
- **Examples**:
  - `0` - Default camera
  - `1` - Second camera
  - `"video.mp4"` - Video file path

### `model_path`
- **Type**: String
- **Default**: `"yolo11n.pt"`
- **Description**: YOLO model file path
- **Options**:
  - `yolo11n.pt` - Nano model (fastest)
  - `yolo11s.pt` - Small model
  - `yolo11m.pt` - Medium model
  - `yolo11l.pt` - Large model
  - `yolo11x.pt` - Extra Large model (most accurate)

### `confidence_threshold`
- **Type**: Float
- **Default**: `0.5`
- **Range**: `0.0 - 1.0`
- **Description**: Detection confidence threshold, higher values mean stricter detection
- **Recommended**: `0.3 - 0.7`

---

## 📹 Camera Settings (`camera_settings`)

### `width` / `height`
- **Type**: Integer
- **Default**: `1280` / `720`
- **Description**: Camera resolution
- **Common combinations**:
  - `640 × 480` - Standard resolution
  - `1280 × 720` - HD resolution
  - `1920 × 1080` - Full HD resolution

### `fps`
- **Type**: Integer
- **Default**: `60`
- **Description**: Camera frame rate
- **Recommended**: `30` or `60`
- **Note**: Higher frame rates may impact performance

### `auto_exposure`
- **Type**: Boolean
- **Default**: `true`
- **Description**: Automatic exposure control
- **Options**:
  - `true` - Automatic exposure adjustment
  - `false` - Manual exposure control

### `brightness` / `contrast` / `saturation`
- **Type**: Integer
- **Default**: `0`
- **Range**: `-100` to `100`
- **Description**: Brightness, contrast, and saturation adjustments
- **Note**: `0` is default, positive values enhance, negative values reduce

---

## 🎨 UI Display Toggles

### `show_ui`
- **Description**: Master switch to control all UI elements

### `show_detection_boxes`
- **Description**: Display YOLO detection boxes (green boxes)

### `show_tracking_boxes`
- **Description**: Display Kalman filter tracking boxes (colored boxes)

### `show_trajectories`
- **Description**: Display object movement trajectory lines

### `show_velocity_vectors`
- **Description**: Display velocity vector arrows

### `show_future_predictions`
- **Description**: Display future position prediction points

### `show_sift_features`
- **Description**: Display SIFT feature points

### `show_detailed_info`
- **Description**: Display detailed text information (confidence, speed, etc.)

### `show_fps_info`
- **Description**: Display FPS and statistics information

### `show_legend`
- **Description**: Display feature legend

### `show_depth_markers`
- **Description**: Display depth markers (cross marks and corner points)

### `show_depth_legend`
- **Description**: Display depth color coding legend

---

## 🖥️ Monitor Window Settings (`monitor_windows`)

### `main_window` - Main Window
- **`enabled`**: Whether to enable the main window
- **`name`**: Window title
- **`position`**: Window position `[x, y]`
- **`size`**: Window size `[width, height]`

### `detection_window` - Detection Window
- **Description**: Shows only YOLO detection results
- **Recommendation**: Usually set to `false` to save resources

### `tracking_window` - Tracking Window
- **Description**: Shows only Kalman filter tracking results
- **Recommendation**: Usually set to `false` to save resources

---

## 🎯 Tracking Parameters (`tracking_params`)

### `max_disappeared`
- **Default**: `30`
- **Description**: Number of frames after which a disappeared target is deleted
- **Calculation**: 30 frames ≈ 1 second (at 30fps)

### `max_distance`
- **Default**: `100`
- **Description**: Maximum association distance between detection and tracker (pixels)

### `trajectory_length`
- **Default**: `90`
- **Description**: Number of frames to keep in trajectory history
- **Calculation**: 90 frames ≈ 3 seconds (at 30fps)

### `future_prediction_steps`
- **Default**: `15`
- **Description**: Total number of future position prediction steps

### `future_prediction_display_steps`
- **Default**: `3`
- **Description**: Number of future prediction points actually displayed

### `sift_feature_limit`
- **Default**: `10`
- **Description**: Maximum number of SIFT feature points displayed per target

---

## 🎨 Display Parameters (`display_params`)

### `font_scale`
- **Default**: `0.7`
- **Range**: `0.5 - 1.0`
- **Description**: Text display scaling factor

### `line_thickness`
- **Default**: `2`
- **Description**: Thickness of trajectory lines and velocity vectors

### `box_thickness`
- **Default**: `2`
- **Description**: Thickness of detection and tracking box borders

### `text_color`
- **Default**: `[255, 255, 255]`
- **Format**: `[B, G, R]` (Blue, Green, Red)
- **Description**: Text color, white is `[255, 255, 255]`

### `background_alpha`
- **Default**: `0.7`
- **Range**: `0.0 - 1.0`
- **Description**: Background transparency of information boxes

---

## 💾 Output Settings (`output_settings`)

### `save_video`
- **Default**: `false`
- **Description**: Whether to record output video

### `output_path`
- **Default**: `"output.mp4"`
- **Description**: File path and name for saved video

### `fps`
- **Default**: `30`
- **Description**: Frame rate of output video

### `codec`
- **Default**: `"mp4v"`
- **Options**: `mp4v`, `XVID`, `H264`
- **Description**: Video encoding format

---

## ⚡ Performance Settings (`performance`)

### `use_gpu`
- **Default**: `true`
- **Description**: Whether to use GPU acceleration (requires CUDA support)

### `max_fps`
- **Default**: `30`
- **Description**: Limit processing frame rate to control CPU usage

### `skip_frames`
- **Default**: `0`
- **Description**: Number of frames to skip, 0 means no frame skipping

### `resize_input`
- **Default**: `false`
- **Description**: Whether to resize input images to improve performance

### `input_size`
- **Default**: `[640, 640]`
- **Description**: Input image size for YOLO model

---

## 🎮 Keyboard Controls (Runtime)

| Key | Function |
|-----|----------|
| `1` | Toggle detection boxes display |
| `2` | Toggle tracking boxes display |
| `3` | Toggle trajectories display |
| `4` | Toggle velocity vectors display |
| `5` | Toggle future predictions display |
| `6` | Toggle SIFT features display |
| `7` | Toggle detailed info display |
| `8` | Toggle FPS info display |
| `9` | Toggle legend display |
| `0` | Toggle depth markers display |
| `-` | Toggle depth legend display |
| `r` | Reset configuration |
| `s` | Save configuration |
| `h` | Show help information |
| `q` | Exit program |

---

## 🌈 Depth Color Coding

| Color | Distance Range | Description |
|-------|----------------|-------------|
| 🔴 Red | < 200px | Very Close |
| 🟠 Orange | 200-500px | Close |
| 🟡 Yellow | 500-1000px | Near |
| 🟢 Green | 1000-2000px | Medium |
| 🔵 Cyan | 2000-3000px | Far |
| 🔵 Blue | > 3000px | Very Far |

---

## 🔧 Configuration Optimization Tips

### 1. Basic Usage
- Modify `camera_id` to select camera or video file
- Adjust `confidence_threshold` to control detection sensitivity
- Choose appropriate `model_path` (n=fastest, x=most accurate)

### 2. Camera Optimization
- **High Quality**: `width=1920, height=1080, fps=30`
- **Balanced Mode**: `width=1280, height=720, fps=30`
- **High Speed Mode**: `width=640, height=480, fps=60`

### 3. Performance Tuning
- **Low-end devices**: `use_gpu=false, max_fps=15, skip_frames=1`
- **High-end devices**: `use_gpu=true, max_fps=60, skip_frames=0`
- **Memory constrained**: `resize_input=true, input_size=[320, 320]`

### 4. Display Adjustments
- **Simplified interface**: Turn off unnecessary `show_*` options
- **Multi-window**: Enable `detection_window` and `tracking_window`
- **Custom colors**: Modify `text_color` and `background_alpha`

### 5. Tracking Optimization
- **Fast movement**: Increase `max_distance` and `future_prediction_steps`
- **Occlusion handling**: Increase `max_disappeared`
- **Trajectory display**: Adjust `trajectory_length`

---

## 🚨 Troubleshooting

### Camera Issues
- **Cannot open camera**: Check `camera_id` and camera connection
- **Unsupported resolution**: Try lowering `width` and `height`

### Performance Issues
- **Low frame rate**: Reduce resolution, enable `skip_frames`
- **Memory shortage**: Set `resize_input=true`

### Detection Issues
- **Inaccurate detection**: Adjust `confidence_threshold`
- **Missing detections**: Lower `confidence_threshold`
- **False detections**: Raise `confidence_threshold`

### GPU Issues
- **CUDA errors**: Set `use_gpu=false`
- **Memory shortage**: Reduce `input_size`

---

## 📁 Related Files

- **Configuration file**: `config.json`
- **Configuration management**: `config.py`
- **GUI interface**: `config_gui.py`
- **Main program**: `yolo_tracker.py`
- **Demo program**: `demo.py`
- **Test scripts**: `quick_test.py`

---

## 🚀 Quick Start

1. **Open configuration tool**:
   ```bash
   python config_gui.py
   ```

2. **Start object tracking**:
   ```bash
   python demo.py
   ```

3. **Test features**:
   ```bash
   python quick_test.py
   ```

4. **Interactive menu**:
   ```bash
   start.bat
   ```

---

## 📞 Technical Support

If you encounter issues, please check:
1. Python version is 3.8+
2. All dependencies are correctly installed
3. CUDA is properly configured (if using GPU)
4. Camera is working properly
5. Configuration file format is correct

Use `python quick_test.py` for quick diagnostics.

---

## 🌟 Enhanced Features

This YOLO11 tracking system includes several advanced features:

### Dual Display System
- **Green Detection Boxes**: Real-time YOLO detection results
- **Colored Tracking Boxes**: Kalman filter tracking with unique colors per object

### Depth Visualization
- **Depth Markers**: Cross marks and corner points with color-coded depth information
- **6-Level Depth Classification**: From very close (red) to very far (blue)

### Advanced Tracking
- **SIFT Feature Matching**: Improved accuracy and reduced ID switching
- **Occlusion Handling**: Continues prediction when detection is lost
- **Future Position Prediction**: Shows predicted movement paths

### Interactive Features
- **Mouse Hover Information**: Detailed stats when hovering over confidence values
- **Real-time Controls**: Toggle features with keyboard shortcuts
- **Multi-window Support**: Separate windows for detection and tracking

### All COCO Classes Support
Detects all 80 COCO dataset classes including:
- People, vehicles, animals
- Furniture, electronics, sports equipment
- Food items, household objects, and more

---

## 📈 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- CPU: Intel i5 or AMD Ryzen 5
- Storage: 2GB free space

### Recommended Requirements
- Python 3.9+
- 16GB+ RAM
- GPU: NVIDIA GTX 1060 or better with CUDA support
- CPU: Intel i7 or AMD Ryzen 7
- Storage: 5GB+ free space

### For High Performance
- 32GB+ RAM
- GPU: NVIDIA RTX 3060 or better
- CPU: Intel i9 or AMD Ryzen 9
- NVMe SSD storage 