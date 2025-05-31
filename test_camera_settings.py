#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for camera settings
"""

import cv2
from config import get_config

def test_camera_settings():
    """Test camera settings configuration"""
    print("=== Testing Camera Settings ===")
    
    config = get_config()
    
    # 顯示當前攝像頭設置
    camera_settings = config.get('camera_settings', {})
    print(f"Current camera settings:")
    print(f"  Resolution: {camera_settings.get('width', 1920)}x{camera_settings.get('height', 1080)}")
    print(f"  FPS: {camera_settings.get('fps', 30)}")
    print(f"  Auto exposure: {camera_settings.get('auto_exposure', True)}")
    print(f"  Brightness: {camera_settings.get('brightness', 0)}")
    print(f"  Contrast: {camera_settings.get('contrast', 0)}")
    print(f"  Saturation: {camera_settings.get('saturation', 0)}")
    
    # 測試攝像頭
    camera_id = config.get('camera_id', 0)
    print(f"\nTesting camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Cannot open camera {camera_id}")
        return False
    
    # 設置攝像頭參數
    width = camera_settings.get('width', 1920)
    height = camera_settings.get('height', 1080)
    fps = camera_settings.get('fps', 30)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # 設置其他參數
    if camera_settings.get('auto_exposure', True):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    brightness = camera_settings.get('brightness', 0)
    contrast = camera_settings.get('contrast', 0)
    saturation = camera_settings.get('saturation', 0)
    
    if brightness != 0:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    if contrast != 0:
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    if saturation != 0:
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
    
    # 獲取實際設置
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nCamera test results:")
    print(f"  Requested resolution: {width}x{height}")
    print(f"  Actual resolution: {actual_width}x{actual_height}")
    print(f"  Requested FPS: {fps}")
    print(f"  Actual FPS: {actual_fps:.1f}")
    
    # 檢查是否成功設置
    resolution_ok = (actual_width == width and actual_height == height)
    fps_ok = abs(actual_fps - fps) < 5  # 允許5fps的誤差
    
    if resolution_ok:
        print("  ✓ Resolution set successfully")
    else:
        print("  ⚠ Resolution setting may not be supported")
    
    if fps_ok:
        print("  ✓ FPS set successfully")
    else:
        print("  ⚠ FPS setting may not be supported")
    
    # 測試讀取幀
    print(f"\nTesting frame capture...")
    ret, frame = cap.read()
    
    if ret:
        print(f"  ✓ Frame captured successfully")
        print(f"  Frame shape: {frame.shape}")
        
        # 顯示幾秒鐘的預覽
        print(f"  Showing 5-second preview... (Press 'q' to quit early)")
        
        start_time = cv2.getTickCount()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 計算實際FPS
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 在幀上顯示信息
            info_text = f"Resolution: {frame.shape[1]}x{frame.shape[0]} | FPS: {current_fps:.1f}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or elapsed_time > 5:
                break
        
        cv2.destroyAllWindows()
        print(f"  Preview completed. Average FPS: {current_fps:.1f}")
        
    else:
        print(f"  ❌ Failed to capture frame")
    
    cap.release()
    
    print(f"\n=== Camera Test Complete ===")
    
    if resolution_ok and fps_ok and ret:
        print("✓ Camera settings test PASSED")
        return True
    else:
        print("⚠ Camera settings test completed with warnings")
        return False

def test_different_resolutions():
    """Test different resolution settings"""
    print("\n=== Testing Different Resolutions ===")
    
    config = get_config()
    camera_id = config.get('camera_id', 0)
    
    resolutions = [
        (1920, 1080),
        (1280, 720),
        (640, 480)
    ]
    
    for width, height in resolutions:
        print(f"\nTesting {width}x{height}...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"  ❌ Cannot open camera")
            continue
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width == width and actual_height == height:
            print(f"  ✓ {width}x{height} supported")
        else:
            print(f"  ⚠ {width}x{height} not supported, got {actual_width}x{actual_height}")
        
        cap.release()

if __name__ == "__main__":
    print("Camera Settings Test Tool")
    print("=" * 50)
    
    # 基本測試
    success = test_camera_settings()
    
    # 解析度測試
    test_different_resolutions()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nTo configure camera settings:")
    print("1. Use GUI: python config_gui.py")
    print("2. Edit config.json directly")
    print("3. Use demo.py with camera") 