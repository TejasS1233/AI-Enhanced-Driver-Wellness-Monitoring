

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


class DataLogger:
    """Logs metrics and session data for analysis"""
    
    def __init__(self, config: dict):
        """
        Initialize data logger
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('DriverWellness.DataLogger')
        
        self.logging_enabled = config['logging']['enabled']
        self.save_metrics = config['logging']['save_metrics']
        
        if not self.logging_enabled:
            self.logger.info("Data logging disabled")
            return
        
        # Create session ID
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup metrics file
        if self.save_metrics:
            self.metrics_file = Path(config['logging']['metrics_file'])
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_metrics_file()
        
        # Session data
        self.session_data = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_frames': 0,
            'frames_with_face': 0,
            'total_alerts': 0,
            'alert_breakdown': {},
            'avg_features': {},
            'session_summary': {}
        }
        
        # Metrics buffer
        self.metrics_buffer = []
        self.buffer_size = 100  # Write to file every N samples
        
        self.logger.info(f"Data Logger initialized - Session: {self.session_id}")
    
    def _initialize_metrics_file(self):
        """Initialize CSV file with headers"""
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'session_id',
                    'frame_number',
                    'face_detected',
                    'left_ear',
                    'right_ear',
                    'avg_ear',
                    'mar',
                    'blink_rate',
                    'head_yaw',
                    'head_pitch',
                    'head_roll',
                    'eyes_closed',
                    'yawning',
                    'head_nodding',
                    'head_turned',
                    'drowsy_probability',
                    'alert_triggered',
                    'alert_type'
                ])
    
    def log_frame_metrics(self, frame_number: int, features: Optional[Dict[str, float]],
                         drowsy_prob: Optional[float] = None,
                         alert_info: Optional[Dict] = None):
        """
        Log metrics for a single frame
        
        Args:
            frame_number: Frame number
            features: Extracted features (None if no face detected)
            drowsy_prob: Drowsiness probability from model
            alert_info: Alert information if triggered
        """
        if not self.logging_enabled or not self.save_metrics:
            return
        
        timestamp = datetime.now().isoformat()
        
        # Update session data
        self.session_data['total_frames'] += 1
        
        if features is not None:
            self.session_data['frames_with_face'] += 1
            
            # Prepare row data
            row = [
                timestamp,
                self.session_id,
                frame_number,
                True,
                features.get('left_ear', 0),
                features.get('right_ear', 0),
                features.get('avg_ear', 0),
                features.get('mar', 0),
                features.get('blink_rate', 0),
                features.get('head_yaw', 0),
                features.get('head_pitch', 0),
                features.get('head_roll', 0),
                features.get('eyes_closed', False),
                features.get('yawning', False),
                features.get('head_nodding', False),
                features.get('head_turned', False),
                drowsy_prob if drowsy_prob is not None else 0,
                alert_info is not None,
                alert_info.get('alert_type', '') if alert_info else ''
            ]
        else:
            # No face detected
            row = [
                timestamp,
                self.session_id,
                frame_number,
                False,
                0, 0, 0, 0, 0, 0, 0, 0,
                False, False, False, False,
                0, False, ''
            ]
        
        # Add to buffer
        self.metrics_buffer.append(row)
        
        # Write to file if buffer is full
        if len(self.metrics_buffer) >= self.buffer_size:
            self._flush_metrics_buffer()
        
        # Log alert if triggered
        if alert_info and alert_info.get('should_alert', False):
            alert_type = alert_info['alert_type']
            self.session_data['total_alerts'] += 1
            
            if alert_type not in self.session_data['alert_breakdown']:
                self.session_data['alert_breakdown'][alert_type] = 0
            self.session_data['alert_breakdown'][alert_type] += 1
    
    def _flush_metrics_buffer(self):
        """Write buffered metrics to file"""
        if not self.metrics_buffer:
            return
        
        try:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.metrics_buffer)
            
            self.metrics_buffer = []
            
        except Exception as e:
            self.logger.error(f"Failed to write metrics to file: {e}")
    
    def calculate_session_summary(self, feature_history: List[Dict[str, float]]):
        """
        Calculate summary statistics for session
        
        Args:
            feature_history: List of feature dictionaries from session
        """
        if not feature_history:
            return
        
        # Calculate averages
        feature_keys = ['avg_ear', 'mar', 'blink_rate', 'head_yaw', 'head_pitch', 'head_roll']
        
        for key in feature_keys:
            values = [f[key] for f in feature_history if key in f]
            if values:
                self.session_data['avg_features'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # Detection rate
        if self.session_data['total_frames'] > 0:
            detection_rate = (self.session_data['frames_with_face'] / 
                            self.session_data['total_frames']) * 100
            self.session_data['session_summary']['face_detection_rate'] = f"{detection_rate:.1f}%"
        
        # Alert rate
        if self.session_data['frames_with_face'] > 0:
            alert_rate = (self.session_data['total_alerts'] / 
                        self.session_data['frames_with_face']) * 100
            self.session_data['session_summary']['alert_rate'] = f"{alert_rate:.2f}%"
    
    def end_session(self, feature_history: Optional[List[Dict[str, float]]] = None):
        """
        End logging session and save summary
        
        Args:
            feature_history: Optional list of all features from session
        """
        if not self.logging_enabled:
            return
        
        # Flush remaining metrics
        if self.save_metrics:
            self._flush_metrics_buffer()
        
        # Update end time
        self.session_data['end_time'] = datetime.now().isoformat()
        
        # Calculate session duration
        start = datetime.fromisoformat(self.session_data['start_time'])
        end = datetime.fromisoformat(self.session_data['end_time'])
        duration = (end - start).total_seconds()
        self.session_data['session_summary']['duration_seconds'] = duration
        
        # Calculate summary if features provided
        if feature_history:
            self.calculate_session_summary(feature_history)
        
        # Save session summary
        summary_file = Path(self.config['logging']['session_logs']) / f'session_{self.session_id}.json'
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            self.logger.info(f"Session summary saved: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session summary: {e}")
        
        # Print summary to console
        self._print_session_summary()
    
    def _print_session_summary(self):
        """Print session summary to console"""
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session_data['session_id']}")
        print(f"Duration: {self.session_data['session_summary'].get('duration_seconds', 0):.1f} seconds")
        print(f"Total Frames: {self.session_data['total_frames']}")
        print(f"Frames with Face: {self.session_data['frames_with_face']}")
        print(f"Face Detection Rate: {self.session_data['session_summary'].get('face_detection_rate', 'N/A')}")
        print(f"\nTotal Alerts: {self.session_data['total_alerts']}")
        
        if self.session_data['alert_breakdown']:
            print("Alert Breakdown:")
            for alert_type, count in self.session_data['alert_breakdown'].items():
                print(f"  - {alert_type}: {count}")
        
        print(f"\nAlert Rate: {self.session_data['session_summary'].get('alert_rate', 'N/A')}")
        print("="*60 + "\n")
    
    def get_session_data(self) -> Dict:
        """Get current session data"""
        return self.session_data.copy()
