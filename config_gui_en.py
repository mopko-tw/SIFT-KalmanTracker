#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Management GUI Tool
Provides graphical interface for managing tracking system configuration
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from config import get_config

class ConfigGUI:
    """Configuration Management GUI"""
    
    def __init__(self):
        self.config = get_config()
        self.root = tk.Tk()
        self.root.title("YOLO Tracking System Configuration Manager")
        self.root.geometry("800x600")
        
        self.setup_ui()
        self.load_config_to_ui()
    
    def setup_ui(self):
        """Setup UI interface"""
        # Create notebook widget
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Basic settings tab
        self.setup_basic_tab(notebook)
        
        # Camera settings tab
        self.setup_camera_tab(notebook)
        
        # UI settings tab
        self.setup_ui_tab(notebook)
        
        # Window settings tab
        self.setup_window_tab(notebook)
        
        # Tracking parameters tab
        self.setup_tracking_tab(notebook)
        
        # Performance settings tab
        self.setup_performance_tab(notebook)
        
        # Button area
        self.setup_buttons()
    
    def setup_basic_tab(self, notebook):
        """Basic settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Basic Settings")
        
        # Camera settings
        ttk.Label(frame, text="Camera ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.camera_var = tk.IntVar()
        camera_spin = ttk.Spinbox(frame, from_=0, to=10, textvariable=self.camera_var, width=10)
        camera_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Detect cameras button
        ttk.Button(frame, text="Detect Cameras", command=self.detect_cameras).grid(row=0, column=2, padx=5, pady=5)
        
        # Model path
        ttk.Label(frame, text="YOLO Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar()
        model_entry = ttk.Entry(frame, textvariable=self.model_var, width=40)
        model_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_model).grid(row=1, column=2, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(frame, text="Confidence Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.conf_var = tk.DoubleVar()
        conf_scale = ttk.Scale(frame, from_=0.1, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL, length=200)
        conf_scale.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        self.conf_label = ttk.Label(frame, text="0.5")
        self.conf_label.grid(row=2, column=2, padx=5, pady=5)
        conf_scale.configure(command=self.update_conf_label)
    
    def setup_camera_tab(self, notebook):
        """Camera settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Camera Settings")
        
        # Resolution settings
        ttk.Label(frame, text="Width:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.camera_width_var = tk.IntVar()
        ttk.Entry(frame, textvariable=self.camera_width_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(frame, text="Height:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.camera_height_var = tk.IntVar()
        ttk.Entry(frame, textvariable=self.camera_height_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Common resolution quick settings
        ttk.Label(frame, text="Common Resolutions:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        resolution_frame = ttk.Frame(frame)
        resolution_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        resolutions = [
            ("1920x1080", 1920, 1080),
            ("1280x720", 1280, 720),
            ("640x480", 640, 480)
        ]
        
        for i, (text, w, h) in enumerate(resolutions):
            ttk.Button(resolution_frame, text=text, 
                      command=lambda w=w, h=h: self.set_resolution(w, h)).pack(side=tk.LEFT, padx=2)
        
        # FPS settings
        ttk.Label(frame, text="FPS:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.camera_fps_var = tk.IntVar()
        ttk.Scale(frame, from_=10, to=60, variable=self.camera_fps_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.camera_fps_var).grid(row=2, column=2, padx=5, pady=5)
        
        # Auto exposure
        self.auto_exposure_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Auto Exposure", variable=self.auto_exposure_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Brightness adjustment
        ttk.Label(frame, text="Brightness:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.brightness_var = tk.IntVar()
        ttk.Scale(frame, from_=-100, to=100, variable=self.brightness_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=4, column=1, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.brightness_var).grid(row=4, column=2, padx=5, pady=5)
        
        # Contrast adjustment
        ttk.Label(frame, text="Contrast:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.contrast_var = tk.IntVar()
        ttk.Scale(frame, from_=-100, to=100, variable=self.contrast_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=5, column=1, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.contrast_var).grid(row=5, column=2, padx=5, pady=5)
        
        # Saturation adjustment
        ttk.Label(frame, text="Saturation:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.saturation_var = tk.IntVar()
        ttk.Scale(frame, from_=-100, to=100, variable=self.saturation_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=6, column=1, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.saturation_var).grid(row=6, column=2, padx=5, pady=5)
        
        # Test camera button
        ttk.Button(frame, text="Test Camera Settings", command=self.test_camera_settings).grid(row=7, column=0, columnspan=2, padx=5, pady=10)
    
    def set_resolution(self, width, height):
        """Set resolution"""
        self.camera_width_var.set(width)
        self.camera_height_var.set(height)
    
    def test_camera_settings(self):
        """Test camera settings"""
        try:
            import cv2
            camera_id = self.camera_var.get()
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                messagebox.showerror("Error", f"Cannot open camera {camera_id}")
                return
            
            # Set parameters
            width = self.camera_width_var.get()
            height = self.camera_height_var.get()
            fps = self.camera_fps_var.get()
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Get actual settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            cap.release()
            
            message = f"Camera Settings Test Results:\n"
            message += f"Requested Resolution: {width}x{height}\n"
            message += f"Actual Resolution: {actual_width}x{actual_height}\n"
            message += f"Requested FPS: {fps}\n"
            message += f"Actual FPS: {actual_fps:.1f}\n"
            
            if actual_width == width and actual_height == height:
                message += "\n✓ Resolution setting successful"
            else:
                message += "\n⚠ Resolution setting may not be supported"
            
            messagebox.showinfo("Camera Test", message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera test failed: {e}")
    
    def setup_ui_tab(self, notebook):
        """UI settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="UI Settings")
        
        # UI toggles
        self.ui_vars = {}
        ui_options = [
            ('show_ui', 'Show UI'),
            ('show_detection_boxes', 'Detection Boxes'),
            ('show_tracking_boxes', 'Tracking Boxes'),
            ('show_trajectories', 'Trajectories'),
            ('show_velocity_vectors', 'Velocity Vectors'),
            ('show_future_predictions', 'Future Predictions'),
            ('show_sift_features', 'SIFT Features'),
            ('show_detailed_info', 'Detailed Info'),
            ('show_fps_info', 'FPS Info'),
            ('show_legend', 'Legend'),
            ('show_depth_markers', 'Depth Markers'),
            ('show_depth_legend', 'Depth Legend')
        ]
        
        for i, (key, text) in enumerate(ui_options):
            self.ui_vars[key] = tk.BooleanVar()
            ttk.Checkbutton(frame, text=text, variable=self.ui_vars[key]).grid(
                row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
    
    def setup_window_tab(self, notebook):
        """Window settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Window Settings")
        
        self.window_vars = {}
        windows = ['main_window', 'detection_window', 'tracking_window']
        
        for i, window in enumerate(windows):
            # Window frame
            window_frame = ttk.LabelFrame(frame, text=f"{window.replace('_', ' ').title()}")
            window_frame.grid(row=i, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
            
            self.window_vars[window] = {}
            
            # Enable toggle
            self.window_vars[window]['enabled'] = tk.BooleanVar()
            ttk.Checkbutton(window_frame, text="Enable", 
                          variable=self.window_vars[window]['enabled']).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            
            # Position settings
            ttk.Label(window_frame, text="Position X:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.window_vars[window]['pos_x'] = tk.IntVar()
            ttk.Entry(window_frame, textvariable=self.window_vars[window]['pos_x'], width=10).grid(row=1, column=1, padx=5, pady=2)
            
            ttk.Label(window_frame, text="Position Y:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
            self.window_vars[window]['pos_y'] = tk.IntVar()
            ttk.Entry(window_frame, textvariable=self.window_vars[window]['pos_y'], width=10).grid(row=1, column=3, padx=5, pady=2)
            
            # Size settings
            ttk.Label(window_frame, text="Width:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            self.window_vars[window]['width'] = tk.IntVar()
            ttk.Entry(window_frame, textvariable=self.window_vars[window]['width'], width=10).grid(row=2, column=1, padx=5, pady=2)
            
            ttk.Label(window_frame, text="Height:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
            self.window_vars[window]['height'] = tk.IntVar()
            ttk.Entry(window_frame, textvariable=self.window_vars[window]['height'], width=10).grid(row=2, column=3, padx=5, pady=2)
    
    def setup_tracking_tab(self, notebook):
        """Tracking parameters tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Tracking Parameters")
        
        # Tracking parameters
        params = [
            ('max_disappeared', 'Max Disappeared Frames', 1, 100),
            ('max_distance', 'Max Distance', 10, 500),
            ('trajectory_length', 'Trajectory Length', 30, 300),
            ('future_prediction_steps', 'Prediction Steps', 5, 50),
            ('future_prediction_display_steps', 'Display Prediction Points', 1, 20),
            ('sift_feature_limit', 'SIFT Feature Limit', 5, 50)
        ]
        
        self.tracking_vars = {}
        for i, (key, text, min_val, max_val) in enumerate(params):
            ttk.Label(frame, text=f"{text}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            self.tracking_vars[key] = tk.IntVar()
            ttk.Scale(frame, from_=min_val, to=max_val, variable=self.tracking_vars[key], 
                     orient=tk.HORIZONTAL, length=200).grid(row=i, column=1, padx=5, pady=5)
            ttk.Label(frame, textvariable=self.tracking_vars[key]).grid(row=i, column=2, padx=5, pady=5)
    
    def setup_performance_tab(self, notebook):
        """Performance settings tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Performance Settings")
        
        # GPU usage
        self.use_gpu_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Use GPU", variable=self.use_gpu_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Max FPS
        ttk.Label(frame, text="Max FPS:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.max_fps_var = tk.IntVar()
        ttk.Scale(frame, from_=10, to=60, variable=self.max_fps_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.max_fps_var).grid(row=1, column=2, padx=5, pady=5)
        
        # Skip frames setting
        ttk.Label(frame, text="Skip Frames:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.skip_frames_var = tk.IntVar()
        ttk.Scale(frame, from_=0, to=10, variable=self.skip_frames_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(frame, textvariable=self.skip_frames_var).grid(row=2, column=2, padx=5, pady=5)
    
    def setup_buttons(self):
        """Setup buttons"""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Config", command=self.reset_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply", command=self.apply_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
    
    def detect_cameras(self):
        """Detect available cameras"""
        cameras = self.config.get_available_cameras()
        if cameras:
            camera_list = ", ".join(map(str, cameras))
            messagebox.showinfo("Available Cameras", f"Found cameras: {camera_list}")
        else:
            messagebox.showwarning("Camera Detection", "No available cameras found")
    
    def browse_model(self):
        """Browse model file"""
        filename = filedialog.askopenfilename(
            title="Select YOLO Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.model_var.set(filename)
    
    def update_conf_label(self, value):
        """Update confidence label"""
        self.conf_label.config(text=f"{float(value):.2f}")
    
    def load_config_to_ui(self):
        """Load configuration to UI"""
        # Basic settings
        self.camera_var.set(self.config.get('camera_id', 0))
        self.model_var.set(self.config.get('model_path', 'yolo11n.pt'))
        self.conf_var.set(self.config.get('confidence_threshold', 0.5))
        
        # Camera settings
        camera_settings = self.config.get('camera_settings', {})
        self.camera_width_var.set(camera_settings.get('width', 1920))
        self.camera_height_var.set(camera_settings.get('height', 1080))
        self.camera_fps_var.set(camera_settings.get('fps', 30))
        self.auto_exposure_var.set(camera_settings.get('auto_exposure', True))
        self.brightness_var.set(camera_settings.get('brightness', 0))
        self.contrast_var.set(camera_settings.get('contrast', 0))
        self.saturation_var.set(camera_settings.get('saturation', 0))
        
        # UI settings
        for key, var in self.ui_vars.items():
            var.set(self.config.get(key, True))
        
        # Window settings
        monitor_windows = self.config.get('monitor_windows', {})
        for window_name, vars_dict in self.window_vars.items():
            window_config = monitor_windows.get(window_name, {})
            vars_dict['enabled'].set(window_config.get('enabled', False))
            position = window_config.get('position', [100, 100])
            vars_dict['pos_x'].set(position[0])
            vars_dict['pos_y'].set(position[1])
            size = window_config.get('size', [640, 480])
            vars_dict['width'].set(size[0])
            vars_dict['height'].set(size[1])
        
        # Tracking parameters
        tracking_params = self.config.get('tracking_params', {})
        for key, var in self.tracking_vars.items():
            var.set(tracking_params.get(key, 30))
        
        # Performance settings
        performance = self.config.get('performance', {})
        self.use_gpu_var.set(performance.get('use_gpu', True))
        self.max_fps_var.set(performance.get('max_fps', 30))
        self.skip_frames_var.set(performance.get('skip_frames', 0))
    
    def load_config(self):
        """Load configuration file"""
        filename = filedialog.askopenfilename(
            title="Load Configuration File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self.config.config = config_data
                self.load_config_to_ui()
                messagebox.showinfo("Success", "Configuration loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def save_config(self):
        """Save configuration file"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration File",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filename:
            try:
                self.apply_config()
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.config.config, f, indent=4, ensure_ascii=False)
                messagebox.showinfo("Success", "Configuration saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def reset_config(self):
        """Reset configuration"""
        if messagebox.askyesno("Confirm", "Are you sure you want to reset all configuration to default values?"):
            self.config.reset_to_default()
            self.load_config_to_ui()
            messagebox.showinfo("Success", "Configuration has been reset to default values")
    
    def apply_config(self):
        """Apply configuration"""
        # Basic settings
        self.config.set('camera_id', self.camera_var.get())
        self.config.set('model_path', self.model_var.get())
        self.config.set('confidence_threshold', self.conf_var.get())
        
        # Camera settings
        self.config.set('camera_settings.width', self.camera_width_var.get())
        self.config.set('camera_settings.height', self.camera_height_var.get())
        self.config.set('camera_settings.fps', self.camera_fps_var.get())
        self.config.set('camera_settings.auto_exposure', self.auto_exposure_var.get())
        self.config.set('camera_settings.brightness', self.brightness_var.get())
        self.config.set('camera_settings.contrast', self.contrast_var.get())
        self.config.set('camera_settings.saturation', self.saturation_var.get())
        
        # UI settings
        for key, var in self.ui_vars.items():
            self.config.set(key, var.get())
        
        # Window settings
        for window_name, vars_dict in self.window_vars.items():
            window_config = {
                'enabled': vars_dict['enabled'].get(),
                'name': window_name.replace('_', ' ').title(),
                'position': [vars_dict['pos_x'].get(), vars_dict['pos_y'].get()],
                'size': [vars_dict['width'].get(), vars_dict['height'].get()]
            }
            self.config.set_window_config(window_name, window_config)
        
        # Tracking parameters
        for key, var in self.tracking_vars.items():
            self.config.set(f'tracking_params.{key}', var.get())
        
        # Performance settings
        self.config.set('performance.use_gpu', self.use_gpu_var.get())
        self.config.set('performance.max_fps', self.max_fps_var.get())
        self.config.set('performance.skip_frames', self.skip_frames_var.get())
        
        self.config.save_config()
        messagebox.showinfo("Success", "Configuration applied and saved")
    
    def run(self):
        """Run GUI"""
        self.root.mainloop()

def main():
    """Main function"""
    app = ConfigGUI()
    app.run()

if __name__ == "__main__":
    main() 