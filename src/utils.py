"""
Utility functions for Driver Wellness Monitoring System
"""

import yaml
import logging
import colorlog
from pathlib import Path
import numpy as np
from datetime import datetime


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise


def setup_logging(config: dict) -> logging.Logger:
    """Setup colorful logging with configuration"""
    log_level = getattr(logging, config['logging']['log_level'])
    
    # Create logger
    logger = logging.getLogger('DriverWellness')
    logger.setLevel(log_level)
    
    # Create console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create colored formatter
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if logging to file is enabled
    if config['logging']['enabled']:
        log_dir = Path(config['logging']['session_logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            log_dir / f'session_{timestamp}.log'
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(point1 - point2)


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle between three points
    p2 is the vertex point
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def normalize_coordinates(landmarks, frame_width: int, frame_height: int) -> np.ndarray:
    """Normalize landmark coordinates to 0-1 range"""
    normalized = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    return normalized


def create_output_dirs(config: dict):
    """Create necessary output directories"""
    dirs = [
        config['logging']['session_logs'],
        'data',
        'models',
        'alerts'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Calculate moving average for smoothing"""
    if len(data) < window_size:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


class RollingBuffer:
    """Circular buffer for storing recent values"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = []
    
    def add(self, value):
        """Add value to buffer"""
        self.buffer.append(value)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
    
    def get_all(self) -> list:
        """Get all values in buffer"""
        return self.buffer
    
    def get_mean(self) -> float:
        """Get mean of buffer values"""
        if not self.buffer:
            return 0.0
        return np.mean(self.buffer)
    
    def get_std(self) -> float:
        """Get standard deviation of buffer values"""
        if len(self.buffer) < 2:
            return 0.0
        return np.std(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) >= self.size
    
    def clear(self):
        """Clear buffer"""
        self.buffer = []


class AlertCooldown:
    """Manage alert cooldown to prevent spam"""
    
    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = {}
    
    def can_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert"""
        current_time = datetime.now().timestamp()
        
        if alert_type not in self.last_alert_time:
            return True
        
        time_since_last = current_time - self.last_alert_time[alert_type]
        return time_since_last >= self.cooldown_seconds
    
    def record_alert(self, alert_type: str):
        """Record that an alert was triggered"""
        self.last_alert_time[alert_type] = datetime.now().timestamp()
    
    def reset(self):
        """Reset all cooldowns"""
        self.last_alert_time = {}
