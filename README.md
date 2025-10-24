# 🚗 AI-Enhanced Driver Wellness Monitoring System

An intelligent real-time system that monitors driver drowsiness and wellness using computer vision and machine learning techniques.

## 📋 Overview

This system uses a cabin-facing camera to detect driver drowsiness, distraction, and stress in real-time through facial landmark analysis and behavioral pattern recognition. It provides timely alerts to prevent accidents caused by driver fatigue.

## 🎯 Features

- **Real-time Face Detection**: 468 facial landmarks using MediaPipe Face Mesh
- **Drowsiness Detection**:
  - Eye Aspect Ratio (EAR) for eye closure detection
  - Blink rate monitoring
  - Prolonged eye closure detection
- **Yawning Detection**: Mouth Aspect Ratio (MAR) analysis
- **Head Pose Estimation**: Detects head nodding and looking away
- **Multi-level Alert System**: Audio and visual alerts with severity levels
- **Data Logging**: Session metrics and analytics
- **Performance Optimized**: Real-time processing at 30 FPS

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or video file
- Windows/Linux/macOS

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/TejasS1233/AI-Enhanced-Driver-Wellness-Monitoring.git
cd AI-Enhanced-Driver-Wellness-Monitoring
```

2. **Create virtual environment** (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**

```powershell
pip install -r requirements.txt
```

## 🚀 Usage

### Run with webcam

```powershell
python src/main_pipeline.py
```

### Run with video file

```powershell
python src/main_pipeline.py --video path/to/video.mp4
```

### Custom configuration

```powershell
python src/main_pipeline.py --config path/to/config.yaml
```

### Controls

- **Q**: Quit application
- **P**: Pause/Resume monitoring
- **R**: Reset system state

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- **Camera settings**: Resolution, FPS, device ID
- **Detection thresholds**: EAR, MAR, head pose limits
- **Alert behavior**: Cooldown periods, escalation
- **Logging**: Metrics saving, session logs
- **Performance**: GPU usage, threading



## 📊 Output

### Real-time Display

- Live video feed with facial landmarks
- EAR and MAR values
- Head pose angles
- Drowsiness probability
- Alert banners

### Data Logging

- `data/metrics.csv`: Frame-by-frame metrics
- `data/logs/session_*.json`: Session summaries
- Alert history and statistics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Tejas S - [TejasS1233](https://github.com/TejasS1233)


---

**⚠️ Safety Notice**: This system is designed as a safety aid and should not replace proper rest and responsible driving practices. Always ensure adequate sleep before driving.
