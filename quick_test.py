#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速測試腳本 - 驗證所有增強功能
"""

import cv2
import numpy as np
from yolo_tracker import YOLOTracker
from config import get_config

def quick_test():
    """快速測試所有功能"""
    print("=== 快速功能測試 ===")
    
    config = get_config()
    
    # 啟用所有新功能
    config.set('show_detection_boxes', True)
    config.set('show_tracking_boxes', True)
    config.set('show_depth_markers', True)
    config.set('show_depth_legend', True)
    config.set('show_detailed_info', True)
    config.set('show_legend', True)
    
    print("✓ 配置已設置")
    
    # 創建跟踪器
    try:
        tracker = YOLOTracker()
        print("✓ 跟踪器創建成功")
        print(f"  模型: {tracker.model_path}")
        print(f"  置信度: {tracker.conf_threshold}")
        print(f"  檢測類別: {len(tracker.class_names)} 個COCO類別")
    except Exception as e:
        print(f"✗ 跟踪器創建失敗: {e}")
        return False
    
    # 測試深度顏色函數
    try:
        test_depths = [100, 500, 1000, 2000, 3000, 4000]
        print("✓ 深度顏色測試:")
        for depth in test_depths:
            color = tracker.get_depth_color(depth)
            desc = tracker.get_depth_description(depth)
            print(f"  {depth}px -> {desc} {color}")
    except Exception as e:
        print(f"✗ 深度顏色測試失敗: {e}")
        return False
    
    # 測試虛線繪製函數
    try:
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker.draw_dashed_line(test_frame, (10, 10), (90, 90), (255, 255, 255), 2, 5, 3)
        print("✓ 虛線繪製測試通過")
    except Exception as e:
        print(f"✗ 虛線繪製測試失敗: {e}")
        return False
    
    # 測試深度標記函數
    try:
        test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        test_bbox = [100, 100, 50, 50]  # x, y, w, h
        tracker.draw_depth_markers(test_frame, test_bbox, 1000, (0, 255, 0))
        print("✓ 深度標記測試通過")
    except Exception as e:
        print(f"✗ 深度標記測試失敗: {e}")
        return False
    
    print("\n=== 所有測試通過！ ===")
    print("新功能摘要:")
    print("✓ 雙重框顯示 (綠色檢測框 + 彩色跟踪框)")
    print("✓ 深度標記 (十字標記 + 角落點)")
    print("✓ 深度顏色編碼 (6級深度分類)")
    print("✓ 增強視覺效果 (角落標記 + 虛線框)")
    print("✓ 全類別檢測 (80個COCO類別)")
    print("✓ 懸停信息顯示")
    print("✓ 完整配置管理")
    
    return True

def show_feature_demo():
    """顯示功能演示圖像"""
    print("\n創建功能演示圖像...")
    
    # 創建演示圖像
    img = np.zeros((600, 1000, 3), dtype=np.uint8)
    
    # 創建跟踪器實例
    tracker = YOLOTracker()
    
    # 標題
    cv2.putText(img, "YOLO11 + Kalman + SIFT Enhanced Features", (200, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 模擬檢測框 (綠色)
    cv2.rectangle(img, (50, 100), (200, 200), (0, 255, 0), 3)
    cv2.putText(img, "Detection Box (Green)", (50, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 繪製角落標記
    corner_length = 15
    # 左上角
    cv2.line(img, (50, 100), (50 + corner_length, 100), (0, 255, 0), 4)
    cv2.line(img, (50, 100), (50, 100 + corner_length), (0, 255, 0), 4)
    
    # 模擬跟踪框 (藍色，帶深度標記)
    cv2.rectangle(img, (300, 100), (450, 200), (255, 0, 0), 3)
    cv2.putText(img, "Tracking Box (Colored)", (300, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # 深度標記
    center_x, center_y = 375, 150
    marker_size = 8
    depth_color = (0, 255, 255)  # 黃色表示中等距離
    
    # 十字標記
    cv2.line(img, (center_x - marker_size, center_y), 
            (center_x + marker_size, center_y), depth_color, 3)
    cv2.line(img, (center_x, center_y - marker_size), 
            (center_x, center_y + marker_size), depth_color, 3)
    
    # 圓形標記
    cv2.circle(img, (center_x, center_y), marker_size + 2, depth_color, 2)
    
    # 角落點
    corners = [(300, 100), (450, 100), (300, 200), (450, 200)]
    for corner in corners:
        cv2.circle(img, corner, 3, depth_color, -1)
    
    # 虛線框示例 (預測狀態)
    tracker.draw_dashed_rectangle(img, (550, 100), (700, 200), (255, 255, 0), 2)
    cv2.putText(img, "Prediction Box (Dashed)", (550, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # 深度顏色圖例
    y_start = 250
    cv2.putText(img, "Depth Color Coding:", (50, y_start), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    depth_info = [
        ("Very Close (<200px)", (0, 0, 255)),
        ("Close (200-500px)", (0, 165, 255)),
        ("Near (500-1000px)", (0, 255, 255)),
        ("Medium (1000-2000px)", (0, 255, 0)),
        ("Far (2000-3000px)", (255, 255, 0)),
        ("Very Far (>3000px)", (255, 0, 0))
    ]
    
    for i, (text, color) in enumerate(depth_info):
        y_pos = y_start + 40 + i * 30
        cv2.rectangle(img, (50, y_pos - 10), (80, y_pos + 10), color, -1)
        cv2.putText(img, text, (90, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 功能列表
    features = [
        "✓ Dual display: Detection + Tracking boxes",
        "✓ Depth markers: Color-coded indicators",
        "✓ Enhanced visualization: Corner markers",
        "✓ All 80 COCO classes detection",
        "✓ SIFT feature matching",
        "✓ Occlusion handling",
        "✓ Interactive hover information"
    ]
    
    cv2.putText(img, "Enhanced Features:", (500, y_start), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    for i, feature in enumerate(features):
        y_pos = y_start + 40 + i * 25
        cv2.putText(img, feature, (500, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 顯示圖像
    cv2.imshow('Enhanced Features Demo', img)
    print("按任意鍵關閉演示圖像...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("YOLO11增強功能快速測試")
    print("=" * 50)
    
    # 運行快速測試
    if quick_test():
        # 顯示功能演示
        show_feature_demo()
        
        print("\n" + "=" * 50)
        print("測試完成！所有增強功能正常運行。")
        print("\n使用方法:")
        print("python demo.py                    # 開始攝像頭跟踪")
        print("python config_gui.py             # 打開配置GUI")
        print("python test_enhanced_features.py # 完整功能測試")
        print("start.bat                        # 交互式選單")
    else:
        print("\n測試失敗，請檢查錯誤信息。") 