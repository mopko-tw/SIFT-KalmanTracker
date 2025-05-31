import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_tracker import KalmanTracker
import cv2

class MultiObjectTracker:
    """多目標跟踪器"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.trackers = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
    def calculate_iou(self, box1, box2):
        """計算IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 轉換為左上角和右下角座標
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        # 計算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
            
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def calculate_distance(self, box1, box2):
        """計算中心點距離"""
        x1, y1 = box1[:2]
        x2, y2 = box2[:2]
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def estimate_depth_from_size(self, bbox, reference_size=100):
        """基於目標大小估計深度"""
        _, _, w, h = bbox
        area = w * h
        if area > 0:
            # 簡單的深度估計：假設目標大小與距離成反比
            depth = reference_size * 1000 / np.sqrt(area)
            return min(max(depth, 50), 5000)  # 限制在50-5000像素範圍
        return 1000  # 默認深度
    
    def match_features(self, desc1, desc2):
        """SIFT特徵匹配"""
        if desc1 is None or desc2 is None:
            return 0.0
            
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # 返回匹配度分數
            if len(desc1) > 0:
                return len(good_matches) / len(desc1)
            return 0.0
            
        except Exception as e:
            print(f"特徵匹配錯誤: {e}")
            return 0.0
    
    def associate_detections_to_trackers(self, detections, trackers, frame=None):
        """關聯檢測和跟踪器"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # 計算成本矩陣
        cost_matrix = np.zeros((len(detections), len(trackers)))
        
        for d, det in enumerate(detections):
            det_bbox = det[:4]
            
            for t, trk in enumerate(trackers):
                trk_bbox = trk['bbox']
                tracker_obj = trk['tracker']
                
                # 基本成本：IoU和距離
                iou = self.calculate_iou(det_bbox, trk_bbox)
                distance = self.calculate_distance(det_bbox, trk_bbox)
                basic_cost = 1 - iou + distance / self.max_distance
                
                # 特徵匹配成本
                feature_cost = 0.0
                if frame is not None and tracker_obj.descriptors is not None:
                    # 提取當前檢測的ROI
                    x, y, w, h = det_bbox
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    
                    # 確保座標在圖像範圍內
                    h_img, w_img = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    
                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            # 提取當前ROI的特徵
                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                            _, desc = self.sift.detectAndCompute(gray, None)
                            
                            # 計算特徵匹配度
                            match_score = self.match_features(desc, tracker_obj.descriptors)
                            feature_cost = 1.0 - match_score
                
                # 組合成本
                cost_matrix[d, t] = 0.7 * basic_cost + 0.3 * feature_cost
        
        # 使用匈牙利算法進行最優匹配
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # 過濾掉成本過高的匹配
        matched_indices = []
        unmatched_detections = []
        unmatched_trackers = []
        
        for d in range(len(detections)):
            if d not in row_indices:
                unmatched_detections.append(d)
        
        for t in range(len(trackers)):
            if t not in col_indices:
                unmatched_trackers.append(t)
        
        for d, t in zip(row_indices, col_indices):
            if cost_matrix[d, t] > 0.7:  # 閾值
                unmatched_detections.append(d)
                unmatched_trackers.append(t)
            else:
                matched_indices.append([d, t])
        
        if len(matched_indices) == 0:
            matched_indices = np.empty((0, 2), dtype=int)
        else:
            matched_indices = np.array(matched_indices)
        
        return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def update(self, detections, frame=None):
        """更新跟踪器"""
        # 預測所有跟踪器的下一個狀態
        predicted_states = []
        tracker_objects = []
        
        for track_id, tracker in list(self.trackers.items()):
            predicted_state = tracker.predict()
            if tracker.is_valid():
                predicted_states.append({
                    'bbox': predicted_state,
                    'tracker': tracker
                })
                tracker_objects.append(track_id)
            else:
                # 移除無效的跟踪器
                del self.trackers[track_id]
        
        # 關聯檢測和跟踪器
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, predicted_states, frame)
        
        # 更新匹配的跟踪器
        for m in matched:
            det_idx, trk_idx = m
            track_id = tracker_objects[trk_idx]
            detection = detections[det_idx]
            
            # 提取ROI圖像
            roi_image = None
            if frame is not None:
                x, y, w, h = detection[:4]
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # 確保座標在圖像範圍內
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if x2 > x1 and y2 > y1:
                    roi_image = frame[y1:y2, x1:x2]
            
            # 更新跟踪器
            confidence = detection[4] if len(detection) > 4 else 0.0
            self.trackers[track_id].update(detection[:4], confidence, roi_image)
            
            # 估計深度
            depth = self.estimate_depth_from_size(detection[:4])
            self.trackers[track_id].update_depth(depth)
        
        # 為未匹配的檢測創建新的跟踪器
        for i in unmatched_dets:
            detection = detections[i]
            confidence = detection[4] if len(detection) > 4 else 0.0
            class_id = int(detection[5]) if len(detection) > 5 else 0
            
            # 提取ROI圖像
            roi_image = None
            if frame is not None:
                x, y, w, h = detection[:4]
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # 確保座標在圖像範圍內
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                if x2 > x1 and y2 > y1:
                    roi_image = frame[y1:y2, x1:x2]
            
            tracker = KalmanTracker(detection[:4], self.next_id, confidence, class_id)
            if roi_image is not None:
                tracker.update(detection[:4], confidence, roi_image)
            
            # 估計初始深度
            depth = self.estimate_depth_from_size(detection[:4])
            tracker.update_depth(depth)
            
            self.trackers[self.next_id] = tracker
            self.next_id += 1
        
        # 返回當前所有有效的跟踪結果
        results = []
        for track_id, tracker in self.trackers.items():
            if tracker.is_valid():
                state = tracker.get_state()
                detailed_info = tracker.get_detailed_info()
                trajectory = tracker.get_trajectory()
                
                results.append({
                    'id': track_id,
                    'bbox': state,
                    'trajectory': trajectory,
                    'time_since_update': tracker.time_since_update,
                    'detailed_info': detailed_info
                })
        
        return results 