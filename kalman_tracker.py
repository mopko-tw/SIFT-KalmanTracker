import numpy as np
from filterpy.kalman import KalmanFilter
from collections import deque
import time
import cv2

class KalmanTracker:
    """卡爾曼濾波跟踪器"""
    
    def __init__(self, bbox, track_id, confidence=0.0, class_id=0):
        self.track_id = track_id
        self.confidence = confidence
        self.class_id = class_id
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 狀態轉移矩陣 [x, y, w, h, vx, vy, vw, vh]
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # 觀測矩陣
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # 過程噪聲協方差
        self.kf.Q = np.eye(8) * 0.1
        
        # 觀測噪聲協方差
        self.kf.R = np.eye(4) * 1.0
        
        # 初始狀態協方差
        self.kf.P *= 10.0
        
        # 初始化狀態
        x, y, w, h = bbox
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0]).reshape(8, 1)
        
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        self.history = deque(maxlen=90)  # 3秒軌跡 (30fps * 3s)
        self.last_detection_time = time.time()
        
        # 深度和特徵相關
        self.depth_history = deque(maxlen=10)
        self.estimated_depth = 0.0
        self.keypoints = None
        self.descriptors = None
        self.roi_image = None
        
        # 運動分析
        self.speed_history = deque(maxlen=30)
        self.direction_history = deque(maxlen=30)
        
    def update(self, bbox, confidence=None, roi_image=None):
        """更新跟踪器"""
        self.time_since_update = 0
        self.history.append(bbox[:2])  # 保存中心點
        self.hit_streak += 1
        self.last_detection_time = time.time()
        
        if confidence is not None:
            self.confidence = confidence
            
        # 保存ROI圖像用於特徵匹配
        if roi_image is not None:
            self.roi_image = roi_image.copy()
            self._extract_features()
        
        # 卡爾曼濾波更新
        self.kf.update(np.array(bbox).reshape(4, 1))
        
        # 更新速度歷史
        velocity = self.get_velocity()
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        self.speed_history.append(speed)
        
        # 更新方向歷史
        if abs(velocity[0]) > 0.1 or abs(velocity[1]) > 0.1:
            direction = np.arctan2(velocity[1], velocity[0]) * 180 / np.pi
            self.direction_history.append(direction)
        
    def _extract_features(self):
        """提取SIFT特徵點"""
        if self.roi_image is None:
            return
            
        try:
            # 創建SIFT檢測器
            sift = cv2.SIFT_create()
            
            # 轉換為灰度圖
            if len(self.roi_image.shape) == 3:
                gray = cv2.cvtColor(self.roi_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.roi_image
                
            # 檢測特徵點和描述符
            self.keypoints, self.descriptors = sift.detectAndCompute(gray, None)
            
        except Exception as e:
            print(f"特徵提取錯誤: {e}")
            self.keypoints = None
            self.descriptors = None
        
    def predict(self):
        """預測下一個狀態"""
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # 返回預測的邊界框
        state = self.kf.x.flatten()
        return [state[0], state[1], state[2], state[3]]
    
    def get_state(self):
        """獲取當前狀態"""
        state = self.kf.x.flatten()
        return [state[0], state[1], state[2], state[3]]
    
    def get_velocity(self):
        """獲取速度"""
        state = self.kf.x.flatten()
        return [state[4], state[5]]  # vx, vy
    
    def get_average_speed(self):
        """獲取平均速度"""
        if len(self.speed_history) == 0:
            return 0.0
        return np.mean(list(self.speed_history))
    
    def get_average_direction(self):
        """獲取平均方向"""
        if len(self.direction_history) == 0:
            return 0.0
        
        # 處理角度的循環性質
        directions = np.array(list(self.direction_history))
        x = np.mean(np.cos(directions * np.pi / 180))
        y = np.mean(np.sin(directions * np.pi / 180))
        return np.arctan2(y, x) * 180 / np.pi
    
    def update_depth(self, depth):
        """更新深度估計"""
        self.depth_history.append(depth)
        if len(self.depth_history) > 0:
            self.estimated_depth = np.mean(list(self.depth_history))
    
    def is_valid(self):
        """檢查跟踪器是否有效"""
        return self.time_since_update < 30  # 1秒內沒有更新就認為無效
    
    def get_trajectory(self):
        """獲取軌跡點"""
        return list(self.history)
    
    def get_detailed_info(self):
        """獲取詳細信息"""
        velocity = self.get_velocity()
        speed = self.get_average_speed()
        direction = self.get_average_direction()
        
        return {
            'confidence': self.confidence,
            'speed': speed,
            'direction': direction,
            'depth': self.estimated_depth,
            'velocity': velocity,
            'age': self.age,
            'hit_streak': self.hit_streak
        } 