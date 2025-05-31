import cv2
import numpy as np
from ultralytics import YOLO
from multi_object_tracker import MultiObjectTracker
import time
import math
from config import get_config

class YOLOTracker:
    """YOLO11 + 卡爾曼濾波跟踪器"""
    
    def __init__(self, model_path=None, conf_threshold=None):
        self.config = get_config()
        
        # 使用配置或參數
        self.model_path = model_path or self.config.get('model_path', 'yolo11n.pt')
        self.conf_threshold = conf_threshold or self.config.get('confidence_threshold', 0.5)
        
        self.model = YOLO(self.model_path)
        self.tracker = MultiObjectTracker(
            max_disappeared=self.config.get('tracking_params.max_disappeared', 30),
            max_distance=self.config.get('tracking_params.max_distance', 100)
        )
        self.colors = self._generate_colors(100)
        
        # YOLO類別名稱
        self.class_names = self.model.names
        
        # 窗口管理
        self.windows = {}
        self._setup_windows()
        
        # 鼠標交互
        self.mouse_pos = (0, 0)
        self.hover_info = None
        self.info_boxes = []  # 存儲信息框位置和內容
        
    def _setup_windows(self):
        """設置監視窗口"""
        monitor_windows = self.config.get('monitor_windows', {})
        
        for window_name, window_config in monitor_windows.items():
            if window_config.get('enabled', False):
                self.windows[window_name] = {
                    'name': window_config.get('name', window_name),
                    'position': window_config.get('position', [100, 100]),
                    'size': window_config.get('size', [640, 480])
                }
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠標回調函數"""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_MOUSEMOVE:
            # 檢查鼠標是否在信息框上
            self.hover_info = None
            for info_box in self.info_boxes:
                x1, y1, x2, y2, info = info_box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.hover_info = info
                    break
        
    def _generate_colors(self, num_colors):
        """生成隨機顏色"""
        colors = []
        for i in range(num_colors):
            hue = i * 180 // num_colors
            color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]))
            colors.append(color)
        return colors
    
    def detect_objects(self, frame):
        """使用YOLO檢測物體"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 獲取邊界框座標 (xyxy格式)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # 轉換為中心點格式 (x, y, w, h)
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append([x_center, y_center, width, height, conf, cls])
        
        return detections
    
    def get_depth_color(self, depth):
        """根據深度獲取顏色"""
        # 深度範圍：50-5000像素
        # 近距離：紅色，中距離：黃色，遠距離：藍色
        if depth < 200:
            # 非常近 - 紅色
            return (0, 0, 255)
        elif depth < 500:
            # 近距離 - 橙色
            return (0, 165, 255)
        elif depth < 1000:
            # 中近距離 - 黃色
            return (0, 255, 255)
        elif depth < 2000:
            # 中距離 - 綠色
            return (0, 255, 0)
        elif depth < 3000:
            # 中遠距離 - 青色
            return (255, 255, 0)
        else:
            # 遠距離 - 藍色
            return (255, 0, 0)
    
    def draw_depth_markers(self, frame, bbox, depth, color):
        """繪製深度標記點"""
        if not self.config.get('show_depth_markers', True):
            return
            
        x, y, w, h = bbox
        center_x, center_y = int(x), int(y)
        
        # 獲取深度顏色
        depth_color = self.get_depth_color(depth)
        
        # 在目標中心繪製深度標記
        marker_size = max(3, min(15, int(w / 20)))
        
        # 繪製十字標記
        cv2.line(frame, (center_x - marker_size, center_y), 
                (center_x + marker_size, center_y), depth_color, 3)
        cv2.line(frame, (center_x, center_y - marker_size), 
                (center_x, center_y + marker_size), depth_color, 3)
        
        # 繪製圓形標記
        cv2.circle(frame, (center_x, center_y), marker_size + 2, depth_color, 2)
        
        # 在目標四角繪製小點
        corner_size = max(2, marker_size // 2)
        corners = [
            (int(x - w/2), int(y - h/2)),  # 左上
            (int(x + w/2), int(y - h/2)),  # 右上
            (int(x - w/2), int(y + h/2)),  # 左下
            (int(x + w/2), int(y + h/2))   # 右下
        ]
        
        for corner in corners:
            cv2.circle(frame, corner, corner_size, depth_color, -1)
    
    def draw_detection_box(self, frame, bbox, conf, cls, color=(0, 255, 0)):
        """繪製檢測框（增強版）"""
        if not self.config.get('show_detection_boxes', True):
            return
            
        x, y, w, h = bbox
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        thickness = self.config.get('display_params.box_thickness', 2)
        
        # 繪製檢測框（實線）
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # 繪製角落標記
        corner_length = min(20, int(w//8), int(h//8))
        corner_thickness = thickness + 1
        
        # 左上角
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # 右上角
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # 左下角
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # 右下角
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # 顯示類別和置信度
        if self.config.get('show_detailed_info', True):
            class_name = self.class_names.get(cls, f'Class{cls}')
            label = f'{class_name}: {conf:.2f}'
            font_scale = self.config.get('display_params.font_scale', 0.7)
            font_thickness = thickness
            
            # 計算文字大小
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # 繪製背景
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # 繪製文字
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       (255, 255, 255), font_thickness)
            
            # 記錄信息框位置和詳細信息
            detailed_info = f'{class_name}: {conf:.3f}'
            self.info_boxes.append((x1, y1 - label_size[1] - 10, 
                                  x1 + label_size[0] + 10, y1, detailed_info))
    
    def get_direction_text(self, direction):
        """將角度轉換為方向文字"""
        if -22.5 <= direction <= 22.5:
            return "→"
        elif 22.5 < direction <= 67.5:
            return "↗"
        elif 67.5 < direction <= 112.5:
            return "↑"
        elif 112.5 < direction <= 157.5:
            return "↖"
        elif 157.5 < direction <= 180 or -180 <= direction <= -157.5:
            return "←"
        elif -157.5 < direction <= -112.5:
            return "↙"
        elif -112.5 < direction <= -67.5:
            return "↓"
        elif -67.5 < direction <= -22.5:
            return "↘"
        return "?"
    
    def draw_prediction_box(self, frame, bbox, track_id, detailed_info, time_since_update, color):
        """繪製預測框和詳細信息（增強版）"""
        if not self.config.get('show_tracking_boxes', True):
            return
            
        x, y, w, h = bbox
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # 根據是否有檢測更新選擇線型
        line_type = cv2.LINE_4 if time_since_update > 0 else cv2.LINE_8
        thickness = 1 if time_since_update > 0 else self.config.get('display_params.box_thickness', 2)
        
        # 繪製預測框（虛線表示預測狀態）
        if time_since_update > 0:
            # 繪製虛線框
            self.draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        else:
            # 繪製實線框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # 繪製深度標記
        depth = detailed_info['depth']
        self.draw_depth_markers(frame, bbox, depth, color)
        
        if not self.config.get('show_detailed_info', True):
            return
            
        # 提取詳細信息
        confidence = detailed_info['confidence']
        speed = detailed_info['speed']
        direction = detailed_info['direction']
        velocity = detailed_info['velocity']
        
        # 狀態文字
        status = "Lost" if time_since_update > 0 else "Track"
        direction_text = self.get_direction_text(direction)
        
        # 獲取深度顏色
        depth_color = self.get_depth_color(depth)
        
        # 顯示ID、置信度和深度
        simple_label = f'ID:{track_id} {confidence:.2f} D:{depth:.0f}'
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.get('display_params.font_scale', 0.7) * 0.8
        font_thickness = 1
        
        # 計算文字大小
        text_size = cv2.getTextSize(simple_label, font, font_scale, font_thickness)[0]
        
        # 繪製信息背景（使用深度顏色）
        info_x = x2 + 5
        info_y = y1
        
        # 確保信息框不超出圖像邊界
        h_img, w_img = frame.shape[:2]
        if info_x + text_size[0] + 10 > w_img:
            info_x = x1 - text_size[0] - 15
        if info_y + 25 > h_img:
            info_y = h_img - 25
        
        # 繪製背景（使用深度顏色的半透明版本）
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), 
                     (info_x + text_size[0] + 10, info_y + 25), 
                     depth_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 繪製邊框
        cv2.rectangle(frame, (info_x, info_y), 
                     (info_x + text_size[0] + 10, info_y + 25), 
                     color, 1)
        
        # 繪製文字
        cv2.putText(frame, simple_label, (info_x + 5, info_y + 18), 
                   font, font_scale, (255, 255, 255), font_thickness)
        
        # 創建詳細信息用於懸停顯示
        detailed_text = (f'ID: {track_id} ({status})\n'
                        f'Confidence: {confidence:.3f}\n'
                        f'Speed: {speed:.1f} px/frame\n'
                        f'Direction: {direction_text} ({direction:.0f}°)\n'
                        f'Depth: {depth:.0f} px ({self.get_depth_description(depth)})\n'
                        f'Velocity: ({velocity[0]:.1f}, {velocity[1]:.1f})')
        
        # 記錄信息框位置和詳細信息
        self.info_boxes.append((info_x, info_y, 
                              info_x + text_size[0] + 10, info_y + 25, 
                              detailed_text))
        
        # 繪製速度向量
        if self.config.get('show_velocity_vectors', True) and speed > 1:
            end_x = int(x + velocity[0] * 15)
            end_y = int(y + velocity[1] * 15)
            cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), color, 2)
            
            # 在箭頭末端顯示速度值
            cv2.putText(frame, f"{speed:.1f}", (end_x + 5, end_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def get_depth_description(self, depth):
        """獲取深度描述"""
        if depth < 200:
            return "Very Close"
        elif depth < 500:
            return "Close"
        elif depth < 1000:
            return "Near"
        elif depth < 2000:
            return "Medium"
        elif depth < 3000:
            return "Far"
        else:
            return "Very Far"
    
    def draw_dashed_rectangle(self, frame, pt1, pt2, color, thickness):
        """繪製虛線矩形"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        dash_length = 10
        gap_length = 5
        
        # 上邊
        self.draw_dashed_line(frame, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)
        # 下邊
        self.draw_dashed_line(frame, (x1, y2), (x2, y2), color, thickness, dash_length, gap_length)
        # 左邊
        self.draw_dashed_line(frame, (x1, y1), (x1, y2), color, thickness, dash_length, gap_length)
        # 右邊
        self.draw_dashed_line(frame, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)
    
    def draw_dashed_line(self, frame, pt1, pt2, color, thickness, dash_length, gap_length):
        """繪製虛線"""
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        # 計算線段長度和方向
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
        
        # 單位向量
        ux = dx / length
        uy = dy / length
        
        # 繪製虛線
        current_length = 0
        while current_length < length:
            # 計算當前段的起點
            start_x = int(x1 + current_length * ux)
            start_y = int(y1 + current_length * uy)
            
            # 計算當前段的終點
            end_length = min(current_length + dash_length, length)
            end_x = int(x1 + end_length * ux)
            end_y = int(y1 + end_length * uy)
            
            # 繪製線段
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
            
            # 移動到下一段
            current_length += dash_length + gap_length
    
    def draw_hover_info(self, frame):
        """繪製懸停詳細信息"""
        if self.hover_info is None:
            return
        
        # 分割多行文字
        lines = self.hover_info.split('\n')
        
        # 計算信息框大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 18
        
        max_width = 0
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            max_width = max(max_width, text_size[0])
        
        # 計算信息框位置（跟隨鼠標）
        info_width = max_width + 20
        info_height = len(lines) * line_height + 10
        
        mouse_x, mouse_y = self.mouse_pos
        info_x = mouse_x + 15
        info_y = mouse_y - info_height - 15
        
        # 確保不超出邊界
        h_img, w_img = frame.shape[:2]
        if info_x + info_width > w_img:
            info_x = mouse_x - info_width - 15
        if info_y < 0:
            info_y = mouse_y + 15
        
        # 繪製半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), 
                     (info_x + info_width, info_y + info_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # 繪製邊框
        cv2.rectangle(frame, (info_x, info_y), 
                     (info_x + info_width, info_y + info_height), 
                     (255, 255, 255), 1)
        
        # 繪製文字
        for i, line in enumerate(lines):
            text_y = info_y + (i + 1) * line_height
            cv2.putText(frame, line, (info_x + 10, text_y), 
                       font, font_scale, (255, 255, 255), font_thickness)
    
    def draw_trajectory(self, frame, trajectory, color):
        """繪製軌跡"""
        if not self.config.get('show_trajectories', True) or len(trajectory) < 2:
            return
        
        # 繪製軌跡線
        points = np.array(trajectory, dtype=np.int32)
        line_thickness = self.config.get('display_params.line_thickness', 2)
        
        for i in range(1, len(points)):
            # 透明度隨時間遞減
            alpha = i / len(points)
            thickness = max(1, int(line_thickness * alpha))
            cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, thickness)
        
        # 繪製軌跡點
        for i, point in enumerate(points):
            alpha = (i + 1) / len(points)
            radius = max(2, int(5 * alpha))
            cv2.circle(frame, tuple(point), radius, color, -1)
    
    def predict_future_position(self, bbox, velocity, steps=None):
        """預測未來位置"""
        if steps is None:
            steps = self.config.get('tracking_params.future_prediction_steps', 15)
            
        x, y, w, h = bbox
        vx, vy = velocity
        
        future_positions = []
        for i in range(1, steps + 1):
            future_x = x + vx * i
            future_y = y + vy * i
            future_positions.append((int(future_x), int(future_y)))
        
        return future_positions
    
    def draw_future_prediction(self, frame, bbox, velocity, color):
        """繪製未來預測位置"""
        if not self.config.get('show_future_predictions', True):
            return
            
        # 獲取總預測步數和顯示步數
        total_steps = self.config.get('tracking_params.future_prediction_steps', 15)
        display_steps = self.config.get('tracking_params.future_prediction_display_steps', 3)
        
        # 確保顯示步數不超過總步數
        display_steps = min(display_steps, total_steps)
        
        future_positions = self.predict_future_position(bbox, velocity, total_steps)
        
        # 只顯示最近期的預測點
        for i in range(display_steps):
            if i < len(future_positions):
                pos = future_positions[i]
                # 透明度隨距離遞減
                alpha = 1.0 - (i / display_steps)
                radius = max(2, int(4 * alpha))
                cv2.circle(frame, pos, radius, color, 1)
    
    def draw_sift_features(self, frame, bbox, tracker, color):
        """繪製SIFT特徵點"""
        if not self.config.get('show_sift_features', True) or tracker.keypoints is None:
            return
            
        x, y, w, h = bbox
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        
        # 在目標區域繪製特徵點
        feature_limit = self.config.get('tracking_params.sift_feature_limit', 10)
        for kp in tracker.keypoints[:feature_limit]:
            pt_x = int(kp.pt[0] + x1)
            pt_y = int(kp.pt[1] + y1)
            cv2.circle(frame, (pt_x, pt_y), 2, color, -1)
    
    def draw_ui_info(self, frame, num_detections, num_tracks, fps, frame_count):
        """繪製UI信息"""
        if not self.config.get('show_fps_info', True):
            return
            
        info_text = [
            f'FPS: {fps:.1f}',
            f'Detections: {num_detections}',
            f'Tracks: {num_tracks}',
            f'Frame: {frame_count}',
            f'SIFT: Enabled'
        ]
        
        font_scale = self.config.get('display_params.font_scale', 0.7)
        text_color = tuple(self.config.get('display_params.text_color', [255, 255, 255]))
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
    
    def draw_legend(self, frame):
        """繪製圖例"""
        if not self.config.get('show_legend', True):
            return
            
        legend_y = 180
        legend_items = [
            ("Green Box: YOLO Detection", (0, 255, 0)),
            ("Color Box: Kalman Tracking", (255, 255, 255)),
            ("Solid Line: Normal Tracking", (255, 255, 255)),
            ("Dashed Line: Occlusion Prediction", (255, 255, 255)),
            ("Arrow: Velocity Vector", (255, 255, 255)),
            ("Dots: SIFT Features", (255, 255, 255)),
            ("Cross Mark: Depth Marker", (255, 255, 255)),
            ("Hover for Details", (255, 255, 0))
        ]
        
        font_scale = self.config.get('display_params.font_scale', 0.7) * 0.7
        
        for i, (text, color) in enumerate(legend_items):
            cv2.putText(frame, text, (10, legend_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        # 繪製深度顏色圖例
        if self.config.get('show_depth_legend', True):
            depth_legend_y = legend_y + len(legend_items) * 20 + 20
            cv2.putText(frame, "Depth Colors:", (10, depth_legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            
            depth_ranges = [
                ("Very Close (<200px)", (0, 0, 255)),
                ("Close (200-500px)", (0, 165, 255)),
                ("Near (500-1000px)", (0, 255, 255)),
                ("Medium (1000-2000px)", (0, 255, 0)),
                ("Far (2000-3000px)", (255, 255, 0)),
                ("Very Far (>3000px)", (255, 0, 0))
            ]
            
            for i, (text, color) in enumerate(depth_ranges):
                y_pos = depth_legend_y + 20 + i * 15
                # 繪製顏色方塊
                cv2.rectangle(frame, (10, y_pos - 8), (25, y_pos + 2), color, -1)
                # 繪製文字
                cv2.putText(frame, text, (30, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), 1)
    
    def create_detection_only_frame(self, frame, detections):
        """創建僅檢測的幀"""
        detection_frame = frame.copy()
        
        for det in detections:
            bbox, conf, cls = det[:4], det[4], det[5]
            self.draw_detection_box(detection_frame, bbox, conf, cls, (0, 255, 0))
        
        return detection_frame
    
    def create_tracking_only_frame(self, frame, tracking_results):
        """創建僅跟踪的幀"""
        tracking_frame = frame.copy()
        
        for result in tracking_results:
            track_id = result['id']
            bbox = result['bbox']
            detailed_info = result['detailed_info']
            trajectory = result['trajectory']
            time_since_update = result['time_since_update']
            
            color = self.colors[track_id % len(self.colors)]
            
            # 繪製預測框和詳細信息
            self.draw_prediction_box(tracking_frame, bbox, track_id, detailed_info, 
                                   time_since_update, color)
            
            # 繪製軌跡
            self.draw_trajectory(tracking_frame, trajectory, color)
            
            # 繪製未來預測
            velocity = detailed_info['velocity']
            self.draw_future_prediction(tracking_frame, bbox, velocity, color)
            
            # 繪製SIFT特徵點
            tracker_obj = self.tracker.trackers.get(track_id)
            if tracker_obj:
                self.draw_sift_features(tracking_frame, bbox, tracker_obj, color)
        
        return tracking_frame
    
    def process_frame(self, frame):
        """處理單幀"""
        # 清空信息框列表
        self.info_boxes = []
        
        # YOLO檢測
        detections = self.detect_objects(frame)
        
        # 提取邊界框用於跟踪
        detection_boxes = []
        for det in detections:
            detection_boxes.append(det)  # 包含完整信息
        
        # 更新跟踪器（傳入原始幀用於特徵提取）
        tracking_results = self.tracker.update(detection_boxes, frame)
        
        # 創建主顯示幀
        main_frame = frame.copy()
        
        # 首先繪製檢測結果（綠色框）
        for det in detections:
            bbox, conf, cls = det[:4], det[4], det[5]
            self.draw_detection_box(main_frame, bbox, conf, cls, (0, 255, 0))
        
        # 然後繪製跟踪結果（彩色框）
        for result in tracking_results:
            track_id = result['id']
            bbox = result['bbox']
            detailed_info = result['detailed_info']
            trajectory = result['trajectory']
            time_since_update = result['time_since_update']
            
            color = self.colors[track_id % len(self.colors)]
            
            # 繪製預測框和詳細信息
            self.draw_prediction_box(main_frame, bbox, track_id, detailed_info, 
                                   time_since_update, color)
            
            # 繪製軌跡
            self.draw_trajectory(main_frame, trajectory, color)
            
            # 繪製未來預測
            velocity = detailed_info['velocity']
            self.draw_future_prediction(main_frame, bbox, velocity, color)
            
            # 繪製SIFT特徵點
            tracker_obj = self.tracker.trackers.get(track_id)
            if tracker_obj:
                self.draw_sift_features(main_frame, bbox, tracker_obj, color)
        
        # 繪製懸停信息
        self.draw_hover_info(main_frame)
        
        return main_frame, detections, tracking_results
    
    def handle_keyboard_input(self, key):
        """處理鍵盤輸入"""
        if key == ord('1'):
            self.config.toggle_ui_element('show_detection_boxes')
            print(f"Detection boxes display: {self.config.get('show_detection_boxes')}")
        elif key == ord('2'):
            self.config.toggle_ui_element('show_tracking_boxes')
            print(f"Tracking boxes display: {self.config.get('show_tracking_boxes')}")
        elif key == ord('3'):
            self.config.toggle_ui_element('show_trajectories')
            print(f"Trajectories display: {self.config.get('show_trajectories')}")
        elif key == ord('4'):
            self.config.toggle_ui_element('show_velocity_vectors')
            print(f"Velocity vectors display: {self.config.get('show_velocity_vectors')}")
        elif key == ord('5'):
            self.config.toggle_ui_element('show_future_predictions')
            print(f"Future predictions display: {self.config.get('show_future_predictions')}")
        elif key == ord('6'):
            self.config.toggle_ui_element('show_sift_features')
            print(f"SIFT features display: {self.config.get('show_sift_features')}")
        elif key == ord('7'):
            self.config.toggle_ui_element('show_detailed_info')
            print(f"Detailed info display: {self.config.get('show_detailed_info')}")
        elif key == ord('8'):
            self.config.toggle_ui_element('show_fps_info')
            print(f"FPS info display: {self.config.get('show_fps_info')}")
        elif key == ord('9'):
            self.config.toggle_ui_element('show_legend')
            print(f"Legend display: {self.config.get('show_legend')}")
        elif key == ord('0'):
            self.config.toggle_ui_element('show_depth_markers')
            print(f"Depth markers display: {self.config.get('show_depth_markers')}")
        elif key == ord('-'):
            self.config.toggle_ui_element('show_depth_legend')
            print(f"Depth legend display: {self.config.get('show_depth_legend')}")
        elif key == ord('r'):
            self.config.reset_to_default()
            print("Configuration reset to default values")
        elif key == ord('s'):
            self.config.save_config()
            print("Configuration saved")
        elif key == ord('h'):
            self.print_help()
    
    def print_help(self):
        """打印幫助信息"""
        print("\n=== Keyboard Controls ===")
        print("1: Toggle detection boxes display")
        print("2: Toggle tracking boxes display")
        print("3: Toggle trajectories display")
        print("4: Toggle velocity vectors display")
        print("5: Toggle future predictions display")
        print("6: Toggle SIFT features display")
        print("7: Toggle detailed info display")
        print("8: Toggle FPS info display")
        print("9: Toggle legend display")
        print("0: Toggle depth markers display")
        print("-: Toggle depth legend display")
        print("r: Reset configuration")
        print("s: Save configuration")
        print("h: Show help")
        print("q: Exit program")
        print("\n=== Mouse Controls ===")
        print("Hover over confidence values to see detailed information")
        print("Move mouse to see tracking details in tooltip")
        print("\n=== Enhanced Features ===")
        print("✓ Dual display: Green detection boxes + Colored tracking boxes")
        print("✓ Depth markers: Color-coded depth indicators with cross marks")
        print("✓ Enhanced visualization: Corner markers and depth colors")
        print("✓ All 80 COCO classes detection (person, car, bicycle, etc.)")
        print("✓ SIFT feature matching for improved accuracy")
        print("✓ Occlusion handling with dashed prediction boxes")
        print("\n=== Depth Color Coding ===")
        print("Red: Very Close (<200px)")
        print("Orange: Close (200-500px)")
        print("Yellow: Near (500-1000px)")
        print("Green: Medium (1000-2000px)")
        print("Cyan: Far (2000-3000px)")
        print("Blue: Very Far (>3000px)")
        print("\n=== Configuration ===")
        print(f"Future prediction points displayed: {self.config.get('tracking_params.future_prediction_display_steps', 3)}")
        print("Use config_gui.py to adjust all settings")
        print("========================\n")
    
    def run_video(self, video_path=None, output_path=None):
        """運行視頻跟踪"""
        # 使用配置中的攝像頭ID
        if video_path is None:
            video_path = self.config.get('camera_id', 0)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Cannot open video source: {video_path}")
            return
        
        # 如果是攝像頭，設置解析度和fps
        if isinstance(video_path, int):
            camera_settings = self.config.get('camera_settings', {})
            
            # 設置解析度
            width = camera_settings.get('width', 1920)
            height = camera_settings.get('height', 1080)
            fps = camera_settings.get('fps', 30)
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            # 設置其他攝像頭參數
            if camera_settings.get('auto_exposure', True):
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 自動曝光
            
            brightness = camera_settings.get('brightness', 0)
            contrast = camera_settings.get('contrast', 0)
            saturation = camera_settings.get('saturation', 0)
            
            if brightness != 0:
                cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            if contrast != 0:
                cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            if saturation != 0:
                cap.set(cv2.CAP_PROP_SATURATION, saturation)
            
            # 驗證設置是否成功
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera resolution set to: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            if actual_width != width or actual_height != height:
                print(f"Warning: Requested {width}x{height}, but got {actual_width}x{actual_height}")
        
        # 設置輸出
        out = None
        if output_path or self.config.get('output_settings.save_video', False):
            output_path = output_path or self.config.get('output_settings.output_path', 'output.mp4')
            fourcc = cv2.VideoWriter_fourcc(*self.config.get('output_settings.codec', 'mp4v'))
            output_fps = self.config.get('output_settings.fps', 30)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            print(f"Output video: {output_path} ({width}x{height} @ {output_fps}fps)")
        
        frame_count = 0
        start_time = time.time()
        
        # 設置窗口和鼠標回調
        main_window_name = None
        for window_name, window_info in self.windows.items():
            cv2.namedWindow(window_info['name'], cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_info['name'], *window_info['size'])
            cv2.moveWindow(window_info['name'], *window_info['position'])
            
            # 設置主窗口的鼠標回調
            if window_name == 'main_window':
                main_window_name = window_info['name']
                cv2.setMouseCallback(main_window_name, self.mouse_callback)
        
        print("Program started. Press 'h' for keyboard controls help")
        print("Hover mouse over confidence values to see detailed information")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 處理幀
            main_frame, detections, tracking_results = self.process_frame(frame)
            
            # 計算FPS
            current_time = time.time()
            fps = frame_count / (current_time - start_time)
            
            # 添加UI信息
            self.draw_ui_info(main_frame, len(detections), len(tracking_results), fps, frame_count)
            self.draw_legend(main_frame)
            
            # 顯示主窗口
            if 'main_window' in self.windows:
                cv2.imshow(self.windows['main_window']['name'], main_frame)
            
            # 顯示檢測窗口
            if 'detection_window' in self.windows:
                detection_frame = self.create_detection_only_frame(frame, detections)
                cv2.imshow(self.windows['detection_window']['name'], detection_frame)
            
            # 顯示跟踪窗口
            if 'tracking_window' in self.windows:
                tracking_frame = self.create_tracking_only_frame(frame, tracking_results)
                cv2.imshow(self.windows['tracking_window']['name'], tracking_frame)
            
            # 保存輸出
            if out:
                out.write(main_frame)
            
            # 處理鍵盤輸入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key != 255:  # 有按鍵按下
                self.handle_keyboard_input(key)
        
        # 清理資源
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # 保存配置
        self.config.save_config()

if __name__ == "__main__":
    # 創建跟踪器
    tracker = YOLOTracker()
    
    # 運行跟踪 (使用配置中的攝像頭)
    tracker.run_video()
    
    # 或者處理視頻文件
    # tracker.run_video('input_video.mp4', 'output_video.mp4') 