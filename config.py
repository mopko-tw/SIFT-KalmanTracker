#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理系統
支持UI開關、攝像頭配置、監視窗口等設置
"""

import json
import os
from typing import Dict, Any
import cv2

class Config:
    """配置管理類"""
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.default_config = {
            # 基本設置
            'camera_id': 0,
            'model_path': 'yolo11n.pt',
            'confidence_threshold': 0.5,
            
            # 攝像頭設置
            'camera_settings': {
                'width': 1920,
                'height': 1080,
                'fps': 30,
                'auto_exposure': True,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0
            },
            
            # UI設置
            'show_ui': True,
            'show_detection_boxes': True,
            'show_tracking_boxes': True,
            'show_trajectories': True,
            'show_velocity_vectors': True,
            'show_future_predictions': True,
            'show_sift_features': True,
            'show_detailed_info': True,
            'show_fps_info': True,
            'show_legend': True,
            'show_depth_markers': True,
            'show_depth_legend': True,
            
            # 監視窗口設置
            'monitor_windows': {
                'main_window': {
                    'enabled': True,
                    'name': 'YOLO11 + Kalman + SIFT Tracking',
                    'position': [100, 100],
                    'size': [1600, 900]
                },
                'detection_window': {
                    'enabled': False,
                    'name': 'Detection Only',
                    'position': [1700, 100],
                    'size': [800, 600]
                },
                'tracking_window': {
                    'enabled': False,
                    'name': 'Tracking Only',
                    'position': [1700, 700],
                    'size': [800, 600]
                }
            },
            
            # 跟踪參數
            'tracking_params': {
                'max_disappeared': 30,
                'max_distance': 100,
                'trajectory_length': 90,
                'future_prediction_steps': 15,
                'future_prediction_display_steps': 3,  # 顯示的未來預測點數量
                'sift_feature_limit': 10
            },
            
            # 顯示參數
            'display_params': {
                'font_scale': 0.7,
                'line_thickness': 2,
                'box_thickness': 2,
                'text_color': [255, 255, 255],
                'background_alpha': 0.7
            },
            
            # 輸出設置
            'output_settings': {
                'save_video': False,
                'output_path': 'output.mp4',
                'fps': 30,
                'codec': 'mp4v'
            },
            
            # 性能設置
            'performance': {
                'use_gpu': True,
                'max_fps': 30,
                'skip_frames': 0,
                'resize_input': False,
                'input_size': [640, 640]
            }
        }
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """載入配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合併默認配置和載入的配置
                config = self.default_config.copy()
                self._deep_update(config, loaded_config)
                return config
                
            except Exception as e:
                print(f"載入配置文件失敗: {e}")
                print("使用默認配置")
                return self.default_config.copy()
        else:
            print("配置文件不存在，使用默認配置")
            return self.default_config.copy()
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"保存配置文件失敗: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default=None):
        """獲取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """設置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_available_cameras(self):
        """獲取可用的攝像頭列表"""
        available_cameras = []
        
        for i in range(10):  # 檢查前10個攝像頭
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        
        return available_cameras
    
    def validate_camera(self, camera_id):
        """驗證攝像頭是否可用"""
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False
        except:
            return False
    
    def print_config(self):
        """打印當前配置"""
        print("=== 當前配置 ===")
        print(json.dumps(self.config, indent=2, ensure_ascii=False))
    
    def reset_to_default(self):
        """重置為默認配置"""
        self.config = self.default_config.copy()
        print("配置已重置為默認值")
    
    def toggle_ui_element(self, element: str):
        """切換UI元素顯示狀態"""
        if element in self.config:
            self.config[element] = not self.config[element]
            return self.config[element]
        return None
    
    def get_window_config(self, window_name: str):
        """獲取窗口配置"""
        return self.config['monitor_windows'].get(window_name, {})
    
    def set_window_config(self, window_name: str, config: Dict):
        """設置窗口配置"""
        if 'monitor_windows' not in self.config:
            self.config['monitor_windows'] = {}
        self.config['monitor_windows'][window_name] = config

# 全局配置實例
config = Config()

def get_config():
    """獲取全局配置實例"""
    return config

def save_config_on_exit():
    """程序退出時保存配置"""
    config.save_config()

# 註冊退出時保存配置
import atexit
atexit.register(save_config_on_exit) 