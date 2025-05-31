#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for future prediction display configuration
"""

from config import get_config
import time

def test_prediction_config():
    """Test the future prediction display configuration"""
    print("=== Testing Future Prediction Display Configuration ===")
    
    config = get_config()
    
    # 顯示當前配置
    current_display_steps = config.get('tracking_params.future_prediction_display_steps', 3)
    total_steps = config.get('tracking_params.future_prediction_steps', 15)
    
    print(f"Current configuration:")
    print(f"  Total prediction steps: {total_steps}")
    print(f"  Display prediction steps: {current_display_steps}")
    
    # 測試不同的配置值
    test_values = [1, 3, 5, 10]
    
    print(f"\nTesting different display step values:")
    for value in test_values:
        config.set('tracking_params.future_prediction_display_steps', value)
        actual_value = config.get('tracking_params.future_prediction_display_steps')
        print(f"  Set to {value}, got: {actual_value}")
    
    # 測試邊界值
    print(f"\nTesting boundary values:")
    
    # 測試超過總步數的情況
    config.set('tracking_params.future_prediction_display_steps', 20)
    display_steps = config.get('tracking_params.future_prediction_display_steps')
    print(f"  Set to 20 (> total steps), got: {display_steps}")
    print(f"  Note: The actual display will be limited to min({display_steps}, {total_steps}) = {min(display_steps, total_steps)}")
    
    # 測試最小值
    config.set('tracking_params.future_prediction_display_steps', 1)
    display_steps = config.get('tracking_params.future_prediction_display_steps')
    print(f"  Set to 1 (minimum), got: {display_steps}")
    
    # 恢復默認值
    config.set('tracking_params.future_prediction_display_steps', 3)
    print(f"\nRestored to default value: 3")
    
    # 保存配置
    config.save_config()
    print("Configuration saved")
    
    print("\n=== Configuration Test Complete ===")
    print("You can now run the main program to see the changes:")
    print("  python demo.py")
    print("Or use the GUI configuration tool:")
    print("  python config_gui.py")
    print("Look for 'Future Prediction Display Steps' in the Tracking Parameters tab")

if __name__ == "__main__":
    test_prediction_config() 