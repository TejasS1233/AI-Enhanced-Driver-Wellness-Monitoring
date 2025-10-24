import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict
from pathlib import Path

from camera_capture import CameraCapture, VideoFileCapture
from face_detection import FaceLandmarkDetector
from feature_extraction import FeatureExtractor
from alert_system import AlertSystem
from data_logger import DataLogger
from utils import load_config, setup_logging, create_output_dirs, RollingBuffer


class DriverWellnessMonitor:
    
    def __init__(self, config_path: str = "config/config.yaml", 
                 video_file: Optional[str] = None):
        
        self.config = load_config(config_path)
        
        self.logger = setup_logging(self.config)
        self.logger.info("="*60)
        self.logger.info("Driver Wellness Monitoring System Starting")
        self.logger.info("="*60)
        
        create_output_dirs(self.config)
        
        self.logger.info("Initializing components...")
        
        if video_file:
            self.camera = VideoFileCapture(self.config, video_file)
            self.logger.info(f"Using video file: {video_file}")
        else:
            self.camera = CameraCapture(self.config)
            self.logger.info("Using live camera feed")
        
        self.face_detector = FaceLandmarkDetector(self.config)
        
        self.feature_extractor = FeatureExtractor(self.config)
        
        self.alert_system = AlertSystem(self.config)
        
        self.data_logger = DataLogger(self.config)
        
        self.feature_history = []
        self.max_history_size = (
            self.config['temporal']['window_size_seconds'] * 
            self.config['camera']['fps']
        )
        
        self.ear_buffer = RollingBuffer(size=10)
        self.mar_buffer = RollingBuffer(size=10)
        
        self.frame_count = 0
        self.fps_buffer = RollingBuffer(size=30)
        self.start_time = None
        
        self.is_running = False
        self.paused = False
        
        self.logger.info("All components initialized successfully")
    
    def start(self):
        if not self.camera.open_camera():
            self.logger.error("Failed to open camera/video")
            return False
        
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info("Monitoring system started")
        self.logger.info("Press 'q' to quit, 'p' to pause, 'r' to reset")
        
        return True
    
    def process_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame_start_time = time.time()
        
        ret, original_frame, preprocessed_frame = self.camera.get_frame()
        
        if not ret or original_frame is None:
            return False, None
        
        self.frame_count += 1
        current_timestamp = time.time() - self.start_time
        
        landmarks = self.face_detector.detect_landmarks(preprocessed_frame)
        
        features = None
        drowsy_prob = None
        alert_decision = None
        
        if landmarks is not None:
            left_eye, right_eye = self.face_detector.extract_eye_landmarks(landmarks)
            mouth = self.face_detector.extract_mouth_landmarks(landmarks)
            head_pose_points = self.face_detector.extract_head_pose_landmarks(landmarks)
            
            features = self.feature_extractor.extract_all_features(
                landmarks=landmarks,
                left_eye=left_eye,
                right_eye=right_eye,
                mouth=mouth,
                head_pose_points=head_pose_points,
                frame_shape=preprocessed_frame.shape,
                timestamp=current_timestamp
            )
            
            self.feature_history.append(features)
            if len(self.feature_history) > self.max_history_size:
                self.feature_history.pop(0)
            
            self.ear_buffer.add(features['avg_ear'])
            self.mar_buffer.add(features['mar'])
            
            is_drowsy = self.feature_extractor.is_drowsy(features)
            
            drowsy_prob = self._estimate_drowsiness_probability(features)
            
            alert_decision = self.alert_system.check_drowsiness(
                features=features,
                drowsy_probability=drowsy_prob
            )
            
            if alert_decision['should_alert']:
                self.alert_system.trigger_alert(alert_decision)
            
            display_frame = self._draw_visualizations(
                original_frame, landmarks, features, drowsy_prob, alert_decision
            )
        else:
            display_frame = original_frame.copy()
            self._draw_no_face_warning(display_frame)
        
        self.data_logger.log_frame_metrics(
            frame_number=self.frame_count,
            features=features,
            drowsy_prob=drowsy_prob,
            alert_info=alert_decision
        )
        
        frame_time = time.time() - frame_start_time
        fps = 1.0 / (frame_time + 1e-6)
        self.fps_buffer.add(fps)
        
        self._draw_info_overlay(display_frame)
        
        return True, display_frame
    
    def _estimate_drowsiness_probability(self, features: Dict[str, float]) -> float:
        score = 0.0
        
        ear_threshold = self.config['features']['ear_threshold']
        if features['avg_ear'] < ear_threshold:
            ear_score = (ear_threshold - features['avg_ear']) / ear_threshold
            score += ear_score * 0.4
        
        mar_threshold = self.config['features']['mar_threshold']
        if features['mar'] > mar_threshold:
            mar_score = min((features['mar'] - mar_threshold) / mar_threshold, 1.0)
            score += mar_score * 0.2
        
        pitch_threshold = self.config['features']['head_pose_pitch_threshold']
        if abs(features['head_pitch']) > pitch_threshold:
            pitch_score = min(abs(features['head_pitch']) / pitch_threshold, 1.0)
            score += pitch_score * 0.3
        
        if features['blink_rate'] > self.config['features']['high_blink_rate']:
            blink_score = min(
                (features['blink_rate'] - self.config['features']['high_blink_rate']) / 20.0,
                1.0
            )
            score += blink_score * 0.1
        
        return min(score, 1.0)
    
    def _draw_visualizations(self, frame: np.ndarray, landmarks: np.ndarray,
                            features: Dict[str, float], drowsy_prob: float,
                            alert_decision: Dict) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = self.face_detector.draw_landmarks(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            landmarks,
            draw_eyes=True,
            draw_mouth=True
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        y_offset = 30
        line_height = 25
        
        ear_color = (0, 255, 0) if features['avg_ear'] > self.config['features']['ear_threshold'] else (0, 0, 255)
        cv2.putText(frame, f"EAR: {features['avg_ear']:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        
        y_offset += line_height
        mar_color = (0, 255, 0) if features['mar'] < self.config['features']['mar_threshold'] else (0, 0, 255)
        cv2.putText(frame, f"MAR: {features['mar']:.3f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)
        
        y_offset += line_height
        cv2.putText(frame, f"Blink Rate: {features['blink_rate']:.1f}/min", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += line_height
        cv2.putText(frame, f"Yaw: {features['head_yaw']:.1f}°", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += line_height
        cv2.putText(frame, f"Pitch: {features['head_pitch']:.1f}°", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += line_height + 10
        prob_color = (0, 255, 0)
        if drowsy_prob > 0.7:
            prob_color = (0, 0, 255)
        elif drowsy_prob > 0.4:
            prob_color = (0, 165, 255)
        
        cv2.putText(frame, f"Drowsiness: {drowsy_prob:.1%}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, prob_color, 2)
        
        if alert_decision and alert_decision['should_alert']:
            self._draw_alert_banner(frame, alert_decision)
        
        return frame
    
    def _draw_alert_banner(self, frame: np.ndarray, alert_decision: Dict):
        h, w = frame.shape[:2]
        
        banner_height = 60
        overlay = frame.copy()
        
        severity = alert_decision['severity']
        if severity >= 8:
            color = (0, 0, 255)
        elif severity >= 5:
            color = (0, 165, 255)
        else:
            color = (0, 255, 255)
        
        cv2.rectangle(overlay, (0, 0), (w, banner_height), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        message = alert_decision['message']
        cv2.putText(frame, message, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    def _draw_no_face_warning(self, frame: np.ndarray):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        
        cv2.putText(frame_bgr, "NO FACE DETECTED", (w//2 - 150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        frame[:] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    def _draw_info_overlay(self, frame: np.ndarray):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame[0, 0, 0] > frame[0, 0, 2] else frame
        else:
            frame_bgr = frame
        
        h, w = frame_bgr.shape[:2]
        
        avg_fps = self.fps_buffer.get_mean()
        cv2.putText(frame_bgr, f"FPS: {avg_fps:.1f}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame_bgr, f"Frame: {self.frame_count}", (w - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if frame_bgr is not frame:
            frame[:] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    def run(self):
        if not self.start():
            return
        
        try:
            while self.is_running:
                if not self.paused:
                    success, display_frame = self.process_frame()
                    
                    if not success:
                        self.logger.warning("Failed to process frame")
                        break
                    
                    if display_frame is not None:
                        display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow('Driver Wellness Monitor', display_bgr)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("Quit requested")
                    break
                elif key == ord('p'):
                    self.paused = not self.paused
                    self.logger.info(f"{'Paused' if self.paused else 'Resumed'}")
                elif key == ord('r'):
                    self.reset()
                    self.logger.info("System reset")
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        
        finally:
            self.stop()
    
    def reset(self):
        self.feature_extractor.reset()
        self.alert_system.reset()
        self.feature_history = []
        self.ear_buffer.clear()
        self.mar_buffer.clear()
    
    def stop(self):
        self.logger.info("Stopping monitoring system...")
        
        self.is_running = False
        
        self.data_logger.end_session(self.feature_history)
        
        self.camera.release()
        self.face_detector.close()
        self.alert_system.cleanup()
        
        cv2.destroyAllWindows()
        
        self.logger.info("Monitoring system stopped")
        self.logger.info("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Driver Wellness Monitoring System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (optional, uses camera if not specified)')
    
    args = parser.parse_args()
    
    monitor = DriverWellnessMonitor(
        config_path=args.config,
        video_file=args.video
    )
    
    monitor.run()


if __name__ == '__main__':
    main()
