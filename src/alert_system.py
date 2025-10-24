import logging
from typing import Dict, Optional
import time
from pathlib import Path
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    
from utils import AlertCooldown


class AlertSystem:
    """Manages alerts and interventions for driver wellness"""
    
    def __init__(self, config: dict):
        """
        Initialize alert system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('DriverWellness.AlertSystem')
        
        # Alert settings
        self.alert_config = config['alerts']
        self.cooldown = AlertCooldown(self.alert_config['cooldown_period'])
        
        # Alert states
        self.current_alerts = set()
        self.alert_start_times = {}
        self.escalation_triggered = {}
        
        # Initialize audio system
        self.audio_enabled = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                self.audio_enabled = True
                self.logger.info("Audio alert system initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize audio: {e}")
        else:
            self.logger.warning("pygame not available - audio alerts disabled")
        
        # Alert sounds (to be loaded)
        self.alert_sounds = {}
        self._load_alert_sounds()
        
        # Alert history
        self.alert_history = []
        
        self.logger.info("Alert System initialized")
    
    def _load_alert_sounds(self):
        """Load alert sound files"""
        if not self.audio_enabled:
            return
        
        alerts_dir = Path("alerts")
        alerts_dir.mkdir(exist_ok=True)
        
        # Try to load drowsy alert sound
        if self.alert_config['drowsy']['enabled']:
            sound_file = self.alert_config['drowsy'].get('sound_file')
            if sound_file and Path(sound_file).exists():
                try:
                    self.alert_sounds['drowsy'] = pygame.mixer.Sound(sound_file)
                    self.logger.info(f"Loaded drowsy alert sound: {sound_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to load sound {sound_file}: {e}")
    
    def check_drowsiness(self, features: Dict[str, float], 
                        drowsy_probability: Optional[float] = None) -> Dict[str, any]:
        """
        Check for drowsiness and determine alert level
        
        Args:
            features: Extracted features from current frame
            drowsy_probability: Optional ML model prediction
            
        Returns:
            Dictionary with alert decision
        """
        current_time = time.time()
        alert_decision = {
            'alert_type': None,
            'severity': 0,
            'message': None,
            'should_alert': False,
            'indicators': []
        }
        
        # Check drowsiness indicators
        if drowsy_probability is not None:
            # Use ML model prediction
            if drowsy_probability > self.config['model']['drowsy_threshold']:
                alert_decision['alert_type'] = 'drowsy'
                alert_decision['severity'] = int(drowsy_probability * 10)
                alert_decision['indicators'].append(f'ML Model: {drowsy_probability:.2%}')
        
        # Rule-based checks
        if features['ear_consecutive'] >= self.config['features']['ear_consecutive_frames']:
            alert_decision['alert_type'] = 'drowsy'
            alert_decision['severity'] = max(alert_decision['severity'], 8)
            alert_decision['indicators'].append(
                f"Prolonged eye closure ({features['ear_consecutive']} frames)"
            )
        
        if features['mar_consecutive'] >= self.config['features']['mar_consecutive_frames']:
            if alert_decision['alert_type'] is None:
                alert_decision['alert_type'] = 'drowsy'
            alert_decision['severity'] = max(alert_decision['severity'], 6)
            alert_decision['indicators'].append(
                f"Frequent yawning ({features['mar_consecutive']} frames)"
            )
        
        if features['head_nodding']:
            if alert_decision['alert_type'] is None:
                alert_decision['alert_type'] = 'drowsy'
            alert_decision['severity'] = max(alert_decision['severity'], 7)
            alert_decision['indicators'].append(
                f"Head nodding (pitch: {features['head_pitch']:.1f}°)"
            )
        
        # Check for distraction
        if features['head_turned']:
            alert_decision['alert_type'] = 'distracted'
            alert_decision['severity'] = max(alert_decision['severity'], 5)
            alert_decision['indicators'].append(
                f"Head turned away (yaw: {features['head_yaw']:.1f}°)"
            )
        
        # Check if enough time has passed since last alert
        if alert_decision['alert_type']:
            if self.cooldown.can_alert(alert_decision['alert_type']):
                alert_decision['should_alert'] = True
                alert_decision['message'] = self._get_alert_message(alert_decision['alert_type'])
                
                # Check for escalation
                if self._should_escalate(alert_decision['alert_type'], current_time):
                    alert_decision['severity'] = 10
                    alert_decision['message'] += " IMMEDIATE ACTION REQUIRED!"
        
        return alert_decision
    
    def _get_alert_message(self, alert_type: str) -> str:
        """Get appropriate message for alert type"""
        if alert_type == 'drowsy':
            return self.alert_config['drowsy']['message']
        elif alert_type == 'stressed':
            return self.alert_config['stressed']['message']
        elif alert_type == 'distracted':
            return self.alert_config['distracted']['message']
        return "Please pay attention to the road."
    
    def _should_escalate(self, alert_type: str, current_time: float) -> bool:
        """Check if alert should be escalated"""
        if not self.alert_config['escalation_enabled']:
            return False
        
        if alert_type not in self.alert_start_times:
            self.alert_start_times[alert_type] = current_time
            return False
        
        time_since_start = current_time - self.alert_start_times[alert_type]
        
        if time_since_start >= self.alert_config['escalation_time']:
            if alert_type not in self.escalation_triggered:
                self.escalation_triggered[alert_type] = True
                return True
        
        return False
    
    def trigger_alert(self, alert_decision: Dict[str, any]):
        """
        Trigger alert based on decision
        
        Args:
            alert_decision: Alert decision dictionary
        """
        if not alert_decision['should_alert']:
            return
        
        alert_type = alert_decision['alert_type']
        message = alert_decision['message']
        severity = alert_decision['severity']
        
        # Log alert
        self.logger.warning(
            f"ALERT [{alert_type.upper()}] Severity: {severity}/10 - {message}"
        )
        
        for indicator in alert_decision['indicators']:
            self.logger.info(f"  └─ {indicator}")
        
        # Play audio alert
        if self.audio_enabled and alert_type in self.alert_sounds:
            try:
                self.alert_sounds[alert_type].play()
            except Exception as e:
                self.logger.error(f"Failed to play alert sound: {e}")
        
        # Visual alert (console)
        if self.alert_config[alert_type].get('visual_alert', False):
            self._show_visual_alert(alert_type, message, severity)
        
        # Record alert
        self.cooldown.record_alert(alert_type)
        self.current_alerts.add(alert_type)
        
        # Add to history
        self.alert_history.append({
            'timestamp': time.time(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'indicators': alert_decision['indicators']
        })
    
    def _show_visual_alert(self, alert_type: str, message: str, severity: int):
        """Display visual alert"""
        # Simple console-based visual alert
        border = "=" * 60
        alert_symbol = "⚠️ " if severity < 8 else "🚨 "
        
        print(f"\n{border}")
        print(f"{alert_symbol} {alert_type.upper()} ALERT {alert_symbol}")
        print(f"{message}")
        print(f"Severity: {'█' * severity}{'░' * (10 - severity)} ({severity}/10)")
        print(f"{border}\n")
    
    def clear_alert(self, alert_type: str):
        """Clear specific alert"""
        if alert_type in self.current_alerts:
            self.current_alerts.remove(alert_type)
        
        if alert_type in self.alert_start_times:
            del self.alert_start_times[alert_type]
        
        if alert_type in self.escalation_triggered:
            del self.escalation_triggered[alert_type]
    
    def get_active_alerts(self) -> set:
        """Get currently active alerts"""
        return self.current_alerts.copy()
    
    def get_alert_history(self, last_n: Optional[int] = None) -> list:
        """
        Get alert history
        
        Args:
            last_n: Number of recent alerts to return (None for all)
            
        Returns:
            List of alert events
        """
        if last_n is None:
            return self.alert_history.copy()
        return self.alert_history[-last_n:]
    
    def reset(self):
        """Reset alert system state"""
        self.current_alerts.clear()
        self.alert_start_times.clear()
        self.escalation_triggered.clear()
        self.cooldown.reset()
        self.logger.info("Alert system reset")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.audio_enabled and pygame.mixer.get_init():
            pygame.mixer.quit()
        self.logger.info("Alert system cleaned up")
