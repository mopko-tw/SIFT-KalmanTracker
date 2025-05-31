#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for hover functionality
"""

from yolo_tracker import YOLOTracker
from config import get_config
import cv2

def test_hover_functionality():
    """Test the hover functionality with camera"""
    print("=== Testing Hover Functionality ===")
    print("This will start the camera with hover functionality enabled")
    print("Move your mouse over confidence values to see detailed information")
    print("Press 'q' to quit, 'h' for help")
    print("===================================")
    
    try:
        # Get configuration
        config = get_config()
        
        # Ensure UI is enabled
        config.set('show_ui', True)
        config.set('show_detailed_info', True)
        
        # Create tracker
        tracker = YOLOTracker()
        
        # Run with camera
        tracker.run_video()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hover_functionality() 