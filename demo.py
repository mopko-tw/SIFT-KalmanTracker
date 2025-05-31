#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11 + 卡爾曼濾波 + SIFT特徵匹配 目標跟踪演示程式
支持攝像頭和視頻文件輸入，具備完整的配置管理系統
"""

import argparse
import cv2
from yolo_tracker import YOLOTracker
from config import get_config

def main():
    parser = argparse.ArgumentParser(description='YOLO11 + Kalman + SIFT Object Tracking')
    parser.add_argument('--source', type=str, default=None, 
                       help='Video source (camera ID or video file path)')
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera ID (0, 1, 2, ...)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--model', type=str, default=None,
                       help='YOLO model path (yolo11n.pt, yolo11s.pt, etc.)')
    parser.add_argument('--conf', type=float, default=None,
                       help='Detection confidence threshold (0.1-1.0)')
    parser.add_argument('--no-ui', action='store_true',
                       help='Disable UI display')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available cameras')
    parser.add_argument('--config', action='store_true',
                       help='Show current configuration')
    parser.add_argument('--reset-config', action='store_true',
                       help='Reset configuration to default')
    
    args = parser.parse_args()
    
    # 獲取配置
    config = get_config()
    
    # 處理特殊命令
    if args.list_cameras:
        cameras = config.get_available_cameras()
        if cameras:
            print("Available cameras:")
            for cam_id in cameras:
                print(f"  Camera {cam_id}")
        else:
            print("No cameras found")
        return
    
    if args.config:
        config.print_config()
        return
    
    if args.reset_config:
        config.reset_to_default()
        config.save_config()
        print("Configuration reset to default values")
        return
    
    # 處理UI設置
    if args.no_ui:
        config.set('show_ui', False)
    
    # 確定視頻源
    video_source = None
    if args.source is not None:
        try:
            # 嘗試轉換為整數（攝像頭ID）
            video_source = int(args.source)
        except ValueError:
            # 如果不是整數，則為文件路徑
            video_source = args.source
    elif args.camera is not None:
        video_source = args.camera
    else:
        # 使用配置中的攝像頭
        video_source = config.get('camera_id', 0)
        print(f"Using camera {video_source} from configuration")
        
        # 檢查攝像頭是否可用
        if not config.validate_camera(video_source):
            print(f"Warning: Camera {video_source} may not be available")
            # 嘗試查找可用攝像頭
            available_cameras = config.get_available_cameras()
            if available_cameras:
                video_source = available_cameras[0]
                print(f"Using camera {video_source} instead")
                config.set('camera_id', video_source)
            else:
                print("No available cameras found")
                return
    
    # 創建跟踪器
    tracker = YOLOTracker(
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    print("=== YOLO11 + Kalman + SIFT Tracking System ===")
    print(f"Video source: {video_source}")
    print(f"Model: {tracker.model_path}")
    print(f"Confidence threshold: {tracker.conf_threshold}")
    print(f"Detection classes: All 80 COCO classes (person, car, bicycle, etc.)")
    print("Enhanced features:")
    print("  ✓ Dual display: Detection boxes (green) + Tracking boxes (colored)")
    print("  ✓ Depth markers: Color-coded depth indicators")
    print("  ✓ Enhanced visualization: Corner markers, depth colors")
    print("  ✓ SIFT feature matching for improved accuracy")
    print("  ✓ Occlusion handling with prediction continuation")
    print("  ✓ Real-time speed and direction analysis")
    print("  ✓ Interactive hover information display")
    print("  ✓ Complete configuration management")
    print("Press 'h' during execution for keyboard controls")
    print("=" * 50)
    
    # 運行跟踪
    try:
        tracker.run_video(video_source, args.output)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 保存配置
        config.save_config()
        print("Configuration saved")

if __name__ == "__main__":
    main() 