#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試增強功能腳本
驗證深度標記、雙重框顯示和更多檢測類別
"""

import cv2
import numpy as np
from yolo_tracker import YOLOTracker
from config import get_config

def test_enhanced_features():
    """測試增強功能"""
    print("=== Testing Enhanced Features ===")
    
    config = get_config()
    
    # 確保所有新功能都啟用
    config.set('show_detection_boxes', True)
    config.set('show_tracking_boxes', True)
    config.set('show_depth_markers', True)
    config.set('show_depth_legend', True)
    config.set('show_detailed_info', True)
    config.set('show_legend', True)
    
    print("Enhanced features enabled:")
    print(f"  ✓ Detection boxes: {config.get('show_detection_boxes')}")
    print(f"  ✓ Tracking boxes: {config.get('show_tracking_boxes')}")
    print(f"  ✓ Depth markers: {config.get('show_depth_markers')}")
    print(f"  ✓ Depth legend: {config.get('show_depth_legend')}")
    print(f"  ✓ Detailed info: {config.get('show_detailed_info')}")
    print(f"  ✓ Legend: {config.get('show_legend')}")
    
    # 創建跟踪器
    tracker = YOLOTracker()
    
    print(f"\nYOLO model: {tracker.model_path}")
    print(f"Confidence threshold: {tracker.conf_threshold}")
    print(f"Available classes: {len(tracker.class_names)} COCO classes")
    
    # 顯示一些主要檢測類別
    important_classes = [
        0,   # person
        1,   # bicycle
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
        14,  # bird
        15,  # cat
        16,  # dog
        17,  # horse
        18,  # sheep
        19,  # cow
        39,  # bottle
        41,  # cup
        56,  # chair
        57,  # couch
        58,  # potted plant
        59,  # bed
        60,  # dining table
        62,  # tv
        63,  # laptop
        64,  # mouse
        65,  # remote
        66,  # keyboard
        67,  # cell phone
    ]
    
    print(f"\nSome detectable objects:")
    for class_id in important_classes[:10]:  # 顯示前10個
        class_name = tracker.class_names.get(class_id, f'Class{class_id}')
        print(f"  {class_id}: {class_name}")
    print(f"  ... and {len(tracker.class_names) - 10} more classes")
    
    # 測試深度顏色映射
    print(f"\nTesting depth color mapping:")
    test_depths = [100, 300, 750, 1500, 2500, 4000]
    for depth in test_depths:
        color = tracker.get_depth_color(depth)
        description = tracker.get_depth_description(depth)
        print(f"  Depth {depth}px: {description} -> Color {color}")
    
    print(f"\nStarting camera test...")
    print("Features to observe:")
    print("  1. Green boxes: YOLO detection results")
    print("  2. Colored boxes: Kalman tracking results")
    print("  3. Cross marks: Depth indicators (color-coded)")
    print("  4. Corner markers: Enhanced box visualization")
    print("  5. Depth legend: Color coding reference")
    print("  6. Hover information: Mouse over confidence values")
    print("  7. All object types: person, car, bicycle, etc.")
    print("\nPress 'h' for keyboard controls")
    print("Press 'q' to exit test")
    print("=" * 50)
    
    # 運行測試
    try:
        tracker.run_video()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        config.save_config()
        print("Test completed, configuration saved")

def create_test_image():
    """創建測試圖像來演示深度顏色"""
    print("\nCreating depth color test image...")
    
    # 創建測試圖像
    img = np.zeros((400, 800, 3), dtype=np.uint8)
    
    # 創建跟踪器實例來使用顏色函數
    tracker = YOLOTracker()
    
    # 測試不同深度的顏色
    depths = [100, 300, 750, 1500, 2500, 4000]
    descriptions = ["Very Close", "Close", "Near", "Medium", "Far", "Very Far"]
    
    for i, (depth, desc) in enumerate(zip(depths, descriptions)):
        color = tracker.get_depth_color(depth)
        
        # 繪製顏色方塊
        x = 50 + i * 120
        y = 150
        cv2.rectangle(img, (x, y), (x + 100, y + 80), color, -1)
        cv2.rectangle(img, (x, y), (x + 100, y + 80), (255, 255, 255), 2)
        
        # 添加文字
        cv2.putText(img, desc, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"{depth}px", (x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 添加標題
    cv2.putText(img, "Depth Color Coding Test", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Press any key to close", (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 顯示圖像
    cv2.imshow('Depth Color Test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Depth color test completed")

if __name__ == "__main__":
    print("Enhanced Features Test Tool")
    print("=" * 50)
    
    # 創建深度顏色測試圖像
    create_test_image()
    
    # 運行增強功能測試
    test_enhanced_features()
    
    print("\n" + "=" * 50)
    print("Enhanced features test completed!")
    print("\nNew features summary:")
    print("✓ Dual display: Detection + Tracking boxes")
    print("✓ Depth markers: Color-coded cross marks")
    print("✓ Enhanced visualization: Corner markers")
    print("✓ All 80 COCO classes detection")
    print("✓ Improved depth color coding")
    print("✓ Interactive hover information") 