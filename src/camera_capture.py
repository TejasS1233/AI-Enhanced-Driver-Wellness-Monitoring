
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

class CameraCapture:
    """Handles camera input and frame preprocessing"""
    
    def __init__(self, config: dict):
        
        self.config = config
        self.logger = logging.getLogger('DriverWellness.Camera')
        
        self.device_id = config['camera']['device_id']
        self.fps = config['camera']['fps']
        self.width = config['camera']['width']
        self.height = config['camera']['height']
        self.target_width = config['preprocessing']['target_width']
        self.target_height = config['preprocessing']['target_height']
        self.apply_clahe = config['preprocessing']['apply_clahe']
        self.normalize = config['preprocessing']['normalize']
        if self.apply_clahe:
            clip_limit = config['preprocessing']['clahe_clip_limit']
            grid_size = tuple(config['preprocessing']['clahe_grid_size'])
            self.clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=grid_size
            )
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.frame_count = 0
    
    def open_camera(self) -> bool:
        
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.device_id}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(
                f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )
            
            self.is_opened = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return ret, frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
       
        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            frame = cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA
            )
        
        if self.apply_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def brightness_normalization(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        v_channel = hsv[:, :, 2]
        mean_brightness = np.mean(v_channel)
        
        target_brightness = 128
        
        if mean_brightness > 0:
            scale = target_brightness / mean_brightness
            hsv[:, :, 2] = np.clip(v_channel * scale, 0, 255).astype(np.uint8)
        
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return frame
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        
        ret, frame = self.read_frame()
        
        if not ret or frame is None:
            return False, None, None
        original = frame.copy()
        preprocessed = self.preprocess_frame(frame)
        
        return True, original, preprocessed
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            self.logger.info("Camera released")
    
    def __del__(self):
        self.release()


class VideoFileCapture(CameraCapture):
    
    def __init__(self, config: dict, video_path: str):
        
        super().__init__(config)
        self.video_path = video_path
    
    def open_camera(self) -> bool:
        
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video file: {self.video_path}")
                return False
            
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(
                f"Video opened: {self.width}x{self.height} @ {self.fps} FPS, "
                f"{total_frames} frames"
            )
            
            self.is_opened = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening video file: {e}")
            return False
