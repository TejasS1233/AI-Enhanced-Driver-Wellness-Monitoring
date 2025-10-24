"""
Face Detection & Landmark Extraction Module
Uses MediaPipe Face Mesh to extract 468 facial landmarks
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict
import logging


class FaceLandmarkDetector:
    """Detects faces and extracts 468 facial landmarks using MediaPipe"""
    
    # MediaPipe Face Mesh landmark indices for specific features
    # Eye landmarks (6 points per eye)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Mouth landmarks (20 points)
    MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                     291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    
    # Head pose landmarks
    NOSE_TIP_INDEX = 1
    CHIN_INDEX = 152
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    
    def __init__(self, config: dict):
        """
        Initialize Face Mesh detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('DriverWellness.FaceDetection')
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh settings
        face_config = config['face_detection']
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=face_config['max_num_faces'],
            min_detection_confidence=face_config['min_detection_confidence'],
            min_tracking_confidence=face_config['min_tracking_confidence'],
            refine_landmarks=face_config['refine_landmarks']
        )
        
        self.logger.info("MediaPipe Face Mesh initialized")
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks in frame
        
        Args:
            frame: Input frame (RGB, normalized 0-1 or 0-255)
            
        Returns:
            Array of shape (468, 3) with landmark coordinates, or None if no face
        """
        # Ensure frame is in correct format for MediaPipe
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
        
        # Process frame
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        h, w = frame.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]  # Convert normalized to pixel coordinates
            for lm in face_landmarks.landmark
        ])
        
        return landmarks
    
    def extract_eye_landmarks(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye landmarks
        
        Args:
            landmarks: All facial landmarks
            
        Returns:
            Tuple of (left_eye_points, right_eye_points)
        """
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        
        return left_eye, right_eye
    
    def extract_mouth_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract mouth landmarks
        
        Args:
            landmarks: All facial landmarks
            
        Returns:
            Mouth landmark points
        """
        mouth = landmarks[self.MOUTH_INDICES]
        return mouth
    
    def extract_head_pose_landmarks(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract landmarks needed for head pose estimation
        
        Args:
            landmarks: All facial landmarks
            
        Returns:
            Dictionary with key points for head pose
        """
        return {
            'nose_tip': landmarks[self.NOSE_TIP_INDEX],
            'chin': landmarks[self.CHIN_INDEX],
            'left_eye': landmarks[self.LEFT_EYE_CORNER],
            'right_eye': landmarks[self.RIGHT_EYE_CORNER],
            'left_mouth': landmarks[self.LEFT_MOUTH_CORNER],
            'right_mouth': landmarks[self.RIGHT_MOUTH_CORNER]
        }
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray,
                      draw_eyes: bool = True, draw_mouth: bool = True) -> np.ndarray:
        """
        Draw landmarks on frame for visualization
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks
            draw_eyes: Whether to highlight eyes
            draw_mouth: Whether to highlight mouth
            
        Returns:
            Frame with drawn landmarks
        """
        # Convert frame to uint8 if needed
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
        
        frame = frame.copy()
        
        # Draw all landmarks as small dots
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        # Highlight eyes
        if draw_eyes:
            left_eye, right_eye = self.extract_eye_landmarks(landmarks)
            
            # Draw eye contours
            left_eye_pts = left_eye[:, :2].astype(np.int32)
            right_eye_pts = right_eye[:, :2].astype(np.int32)
            
            cv2.polylines(frame, [left_eye_pts], True, (255, 0, 0), 2)
            cv2.polylines(frame, [right_eye_pts], True, (255, 0, 0), 2)
        
        # Highlight mouth
        if draw_mouth:
            mouth = self.extract_mouth_landmarks(landmarks)
            mouth_pts = mouth[:, :2].astype(np.int32)
            cv2.polylines(frame, [mouth_pts], True, (0, 0, 255), 2)
        
        return frame
    
    def get_face_bounding_box(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box around detected face
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (x, y, width, height)
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        
        width = x_max - x_min
        height = y_max - y_min
        
        return x_min, y_min, width, height
    
    def is_face_visible(self, landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """
        Check if face is properly visible in frame
        
        Args:
            landmarks: Facial landmarks
            frame_shape: Shape of frame (height, width)
            
        Returns:
            True if face is visible, False otherwise
        """
        h, w = frame_shape[:2]
        
        # Check if key landmarks are within frame bounds
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        # Face should not be too close to edges
        margin = 20
        
        if (np.min(x_coords) < margin or np.max(x_coords) > w - margin or
            np.min(y_coords) < margin or np.max(y_coords) > h - margin):
            return False
        
        # Face should be reasonably sized (not too small)
        face_width = np.max(x_coords) - np.min(x_coords)
        face_height = np.max(y_coords) - np.min(y_coords)
        
        min_face_size = min(w, h) * 0.1
        
        if face_width < min_face_size or face_height < min_face_size:
            return False
        
        return True
    
    def close(self):
        """Release resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
            self.logger.info("Face Mesh detector closed")
