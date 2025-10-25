import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import logging
from pathlib import Path

from camera_capture import CameraCapture
from face_detection import FaceLandmarkDetector
from feature_extraction import FeatureExtractor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataCollection')


class DataCollector:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.camera = CameraCapture(
            device_id=self.config['camera']['device_id'],
            frame_width=self.config['camera']['frame_width'],
            frame_height=self.config['camera']['frame_height'],
            fps=self.config['camera']['fps']
        )
        
        self.face_detector = FaceLandmarkDetector(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.feature_extractor = FeatureExtractor(
            ear_threshold=self.config['features']['ear_threshold'],
            mar_threshold=self.config['features']['mar_threshold'],
            ear_consec_frames=self.config['features']['ear_consec_frames'],
            mar_consec_frames=self.config['features']['mar_consec_frames']
        )
        
        self.collected_data = []
        self.current_label = 0
        self.frame_count = 0
        
        self.output_dir = Path('data')
        self.output_dir.mkdir(exist_ok=True)
    
    def run(self):
        logger.info("="*60)
        logger.info("TRAINING DATA COLLECTION MODE")
        logger.info("="*60)
        logger.info("Controls:")
        logger.info("  [0] - Mark as ALERT (normal driving)")
        logger.info("  [1] - Mark as DROWSY (tired/distracted)")
        logger.info("  [SPACE] - Toggle label (default: ALERT)")
        logger.info("  [S] - Save collected data")
        logger.info("  [Q] - Quit without saving")
        logger.info("="*60)
        
        if not self.camera.start():
            logger.error("Failed to start camera")
            return
        
        try:
            while True:
                frame = self.camera.read_frame()
                if frame is None:
                    continue
                
                landmarks = self.face_detector.detect_landmarks(frame)
                
                if landmarks is not None:
                    features = self.feature_extractor.extract_all_features(
                        landmarks, frame.shape
                    )
                    
                    if features is not None:
                        features['is_drowsy'] = self.current_label
                        features['timestamp'] = datetime.now().isoformat()
                        self.collected_data.append(features)
                        self.frame_count += 1
                        
                        for landmark in landmarks:
                            x = int(landmark[0] * frame.shape[1])
                            y = int(landmark[1] * frame.shape[0])
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                label_text = "DROWSY" if self.current_label == 1 else "ALERT"
                label_color = (0, 0, 255) if self.current_label == 1 else (0, 255, 0)
                
                cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
                cv2.putText(frame, f"Label: {label_text}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                cv2.putText(frame, f"Frames: {self.frame_count}", (20, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(frame, "0=Alert | 1=Drowsy | S=Save | Q=Quit",
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('0'):
                    self.current_label = 0
                    logger.info("Switched to ALERT label")
                
                elif key == ord('1'):
                    self.current_label = 1
                    logger.info("Switched to DROWSY label")
                
                elif key == ord(' '):
                    self.current_label = 1 - self.current_label
                    label_text = "DROWSY" if self.current_label == 1 else "ALERT"
                    logger.info(f"Toggled to {label_text}")
                
                elif key == ord('s') or key == ord('S'):
                    self.save_data()
                
                elif key == ord('q') or key == ord('Q'):
                    logger.info("Quitting without saving...")
                    break
        
        finally:
            self.camera.stop()
            self.face_detector.close()
            cv2.destroyAllWindows()
    
    def save_data(self):
        if not self.collected_data:
            logger.warning("No data to save!")
            return
        
        df = pd.DataFrame(self.collected_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"training_data_{timestamp}.csv"
        
        df.to_csv(output_file, index=False)
        
        logger.info("="*60)
        logger.info(f"Saved {len(df)} samples to {output_file}")
        logger.info(f"  Alert samples: {len(df[df['is_drowsy']==0])}")
        logger.info(f"  Drowsy samples: {len(df[df['is_drowsy']==1])}")
        logger.info("="*60)
        
        self.collected_data = []
        self.frame_count = 0


def merge_datasets():
    data_dir = Path('data')
    csv_files = list(data_dir.glob('training_data_*.csv'))
    
    if not csv_files:
        logger.warning("No training data files found to merge")
        return
    
    logger.info(f"Found {len(csv_files)} training data files")
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        logger.info(f"  {csv_file.name}: {len(df)} samples")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_file = data_dir / 'training_data.csv'
    merged_df.to_csv(output_file, index=False)
    
    logger.info("="*60)
    logger.info(f"Merged dataset saved to {output_file}")
    logger.info(f"  Total samples: {len(merged_df)}")
    logger.info(f"  Alert samples: {len(merged_df[merged_df['is_drowsy']==0])}")
    logger.info(f"  Drowsy samples: {len(merged_df[merged_df['is_drowsy']==1])}")
    logger.info("="*60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'merge':
        merge_datasets()
    else:
        collector = DataCollector()
        collector.run()
