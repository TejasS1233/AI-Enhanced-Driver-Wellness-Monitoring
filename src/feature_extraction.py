"""
Feature Extraction Module
Calculates EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio),
Head Pose (Yaw, Pitch, Roll), and Blink Rate
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from scipy.spatial import distance
import cv2


class FeatureExtractor:
    """Extracts drowsiness-related features from facial landmarks"""
    
    def __init__(self, config: dict):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('DriverWellness.FeatureExtraction')
        
        # Thresholds
        self.ear_threshold = config['features']['ear_threshold']
        self.mar_threshold = config['features']['mar_threshold']
        
        # Counters for consecutive frames
        self.ear_consecutive = 0
        self.mar_consecutive = 0
        
        # Blink detection
        self.blink_counter = 0
        self.blink_timestamps = []
        self.is_blinking = False
        
        # Camera matrix for head pose (will be set based on frame size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        self.logger.info("Feature Extractor initialized")
    
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        where p1-p6 are eye landmarks
        
        Args:
            eye_landmarks: Array of 6 eye landmark points
            
        Returns:
            Eye Aspect Ratio value
        """
        # Vertical eye distances
        vertical_1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        vertical_2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal eye distance
        horizontal = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
        
        return ear
    
    def calculate_mar(self, mouth_landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR)
        
        MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
        
        Args:
            mouth_landmarks: Array of mouth landmark points
            
        Returns:
            Mouth Aspect Ratio value
        """
        # Vertical mouth distances (use subset of landmarks)
        vertical_1 = distance.euclidean(mouth_landmarks[2], mouth_landmarks[10])
        vertical_2 = distance.euclidean(mouth_landmarks[3], mouth_landmarks[9])
        vertical_3 = distance.euclidean(mouth_landmarks[4], mouth_landmarks[8])
        
        # Horizontal mouth distance
        horizontal = distance.euclidean(mouth_landmarks[0], mouth_landmarks[6])
        
        # Calculate MAR
        mar = (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal + 1e-6)
        
        return mar
    
    def detect_blink(self, ear: float, timestamp: float) -> bool:
        """
        Detect blink based on EAR
        
        Args:
            ear: Eye Aspect Ratio
            timestamp: Current timestamp
            
        Returns:
            True if blink detected, False otherwise
        """
        blink_detected = False
        
        # Eye closed
        if ear < self.ear_threshold:
            if not self.is_blinking:
                self.is_blinking = True
        # Eye opened after being closed
        else:
            if self.is_blinking:
                self.is_blinking = False
                self.blink_counter += 1
                self.blink_timestamps.append(timestamp)
                blink_detected = True
                
                # Keep only last 60 seconds of blinks
                cutoff_time = timestamp - 60.0
                self.blink_timestamps = [
                    t for t in self.blink_timestamps if t > cutoff_time
                ]
        
        return blink_detected
    
    def get_blink_rate(self) -> float:
        """
        Calculate blink rate (blinks per minute)
        
        Returns:
            Blink rate in blinks/minute
        """
        if len(self.blink_timestamps) < 2:
            return 0.0
        
        # Calculate time window
        time_window = self.blink_timestamps[-1] - self.blink_timestamps[0]
        
        if time_window < 1.0:
            return 0.0
        
        # Blinks per minute
        blink_rate = (len(self.blink_timestamps) / time_window) * 60.0
        
        return blink_rate
    
    def calculate_head_pose(self, face_landmarks: Dict[str, np.ndarray],
                           frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Calculate head pose angles (yaw, pitch, roll)
        
        Args:
            face_landmarks: Dictionary with key facial landmarks
            frame_shape: Shape of frame (height, width)
            
        Returns:
            Dictionary with yaw, pitch, roll angles in degrees
        """
        h, w = frame_shape[:2]
        
        # Initialize camera matrix if not done
        if self.camera_matrix is None:
            focal_length = w
            center = (w / 2, h / 2)
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
        
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye corner
            (225.0, 170.0, -135.0),    # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        image_points = np.array([
            face_landmarks['nose_tip'][:2],
            face_landmarks['chin'][:2],
            face_landmarks['left_eye'][:2],
            face_landmarks['right_eye'][:2],
            face_landmarks['left_mouth'][:2],
            face_landmarks['right_mouth'][:2]
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        # Extract angles from rotation matrix
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }
    
    def extract_all_features(self, landmarks: np.ndarray,
                            left_eye: np.ndarray,
                            right_eye: np.ndarray,
                            mouth: np.ndarray,
                            head_pose_points: Dict[str, np.ndarray],
                            frame_shape: Tuple[int, int],
                            timestamp: float) -> Dict[str, float]:
        """
        Extract all features from landmarks
        
        Args:
            landmarks: All facial landmarks
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
            mouth: Mouth landmarks
            head_pose_points: Key points for head pose
            frame_shape: Frame dimensions
            timestamp: Current timestamp
            
        Returns:
            Dictionary with all extracted features
        """
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calculate MAR
        mar = self.calculate_mar(mouth)
        
        # Detect blink
        blink_detected = self.detect_blink(avg_ear, timestamp)
        blink_rate = self.get_blink_rate()
        
        # Calculate head pose
        head_pose = self.calculate_head_pose(head_pose_points, frame_shape)
        
        # Check for drowsiness indicators
        eyes_closed = avg_ear < self.ear_threshold
        yawning = mar > self.mar_threshold
        head_nodding = abs(head_pose['pitch']) > self.config['features']['head_pose_pitch_threshold']
        head_turned = abs(head_pose['yaw']) > self.config['features']['head_pose_yaw_threshold']
        
        # Update consecutive frame counters
        if eyes_closed:
            self.ear_consecutive += 1
        else:
            self.ear_consecutive = 0
        
        if yawning:
            self.mar_consecutive += 1
        else:
            self.mar_consecutive = 0
        
        features = {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'mar': mar,
            'blink_detected': blink_detected,
            'blink_rate': blink_rate,
            'head_yaw': head_pose['yaw'],
            'head_pitch': head_pose['pitch'],
            'head_roll': head_pose['roll'],
            'eyes_closed': eyes_closed,
            'yawning': yawning,
            'head_nodding': head_nodding,
            'head_turned': head_turned,
            'ear_consecutive': self.ear_consecutive,
            'mar_consecutive': self.mar_consecutive,
            'timestamp': timestamp
        }
        
        return features
    
    def is_drowsy(self, features: Dict[str, float]) -> bool:
        """
        Determine if driver is drowsy based on features
        
        Args:
            features: Extracted features
            
        Returns:
            True if drowsy indicators detected
        """
        # Drowsiness indicators
        prolonged_eye_closure = features['ear_consecutive'] >= self.config['features']['ear_consecutive_frames']
        frequent_yawning = features['mar_consecutive'] >= self.config['features']['mar_consecutive_frames']
        head_nodding = features['head_nodding']
        
        # High blink rate can indicate fatigue
        high_blink_rate = features['blink_rate'] > self.config['features']['high_blink_rate']
        
        return prolonged_eye_closure or frequent_yawning or head_nodding or high_blink_rate
    
    def reset(self):
        """Reset feature extractor state"""
        self.ear_consecutive = 0
        self.mar_consecutive = 0
        self.blink_counter = 0
        self.blink_timestamps = []
        self.is_blinking = False
