
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

class CameraCapture:
    """Handles camera input and frame preprocessing"""
    
    def __init__(self, config: dict):
        """
        Initialize camera capture with configuration
        
        Args:
            config: Configuration dictionary
        """
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
        """
        Open camera device
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.device_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Verify settings
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
        """
        Read a frame from camera
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return ret, frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for face detection and feature extraction
        
        Args:
            frame: Input frame in RGB format
            
        Returns:
            Preprocessed frame
        """
        # Resize frame
        if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
            frame = cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA
            )
        
        # Apply CLAHE for better contrast in varying lighting
        if self.apply_clahe:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Normalize pixel values to [0, 1]
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def brightness_normalization(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply adaptive brightness normalization
        
        Args:
            frame: Input frame
            
        Returns:
            Brightness-normalized frame
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Get current brightness (V channel)
        v_channel = hsv[:, :, 2]
        mean_brightness = np.mean(v_channel)
        
        # Target brightness
        target_brightness = 128
        
        # Adjust brightness
        if mean_brightness > 0:
            scale = target_brightness / mean_brightness
            hsv[:, :, 2] = np.clip(v_channel * scale, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return frame
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get preprocessed frame from camera
        
        Returns:
            Tuple of (success, original_frame, preprocessed_frame)
        """
        ret, frame = self.read_frame()
        
        if not ret or frame is None:
            return False, None, None
        original = frame.copy()
        preprocessed = self.preprocess_frame(frame)
        
        return True, original, preprocessed
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            self.logger.info("Camera released")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()


class VideoFileCapture(CameraCapture):
    """Handles video file input instead of camera"""
    
    def __init__(self, config: dict, video_path: str):
        """
        Initialize video file capture
        
        Args:
            config: Configuration dictionary
            video_path: Path to video file
        """
        super().__init__(config)
        self.video_path = video_path
    
    def open_camera(self) -> bool:
        """
        Open video file
        
        Returns:
            True if video opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video file: {self.video_path}")
                return False
            
            # Get video properties
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
