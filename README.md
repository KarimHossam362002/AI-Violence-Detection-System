# AI Violence Detection System Using Smart CCTV

## Overview
The **AI Violence Detection System** is an intelligent surveillance solution designed to automatically detect **violent behavior and weapon usage** in real-time CCTV video streams  
The system leverages **deep learning–based spatial and temporal feature extraction** to address the limitations of manual surveillance, including operator fatigue and delayed response times.

This project is developed as a **Graduation Project** and aligns with modern **Smart City** and **Public Safety** initiatives.

---

## Key Objectives
- Automatic detection of violent actions in CCTV footage
- Real-time identification of weapon usage
- Reduction of human dependency in continuous surveillance
- Fast alerting and evidence collection for timely intervention
- Scalable architecture suitable for smart-city deployment

---

## System Architecture
The system is composed of the following main components:

1. **Video Input Layer**
   - Live CCTV streams or recorded video feeds

2. **AI Processing Layer**
   - Frame extraction and preprocessing
   - Violence action recognition (CNN + LSTM / 3D CNN)
   - Weapon detection model (YOLO-based)

3. **Backend Layer (Laravel)**
   - RESTful APIs for alert ingestion
   - Incident and camera management
   - Secure evidence storage

4. **Frontend Dashboard**
   - Real-time alerts and notifications
   - Live camera monitoring
   - Incident review and evidence playback

---

## AI Model Pipeline
- **Spatial Feature Extraction**: CNN-based visual feature learning
- **Temporal Feature Extraction**: Motion modeling using LSTM or 3D CNN
- **Weapon Detection**: Object detection with bounding-box localization
- **Inference Output**:
  - Event type (Violence / Weapon Detected)
  - Confidence score
  - Timestamp and camera metadata

---

## Dataset
- Source: Kaggle + Smart-City CCTV Violence Detection Dataset (SCVD)
- Classes:
  - Violence
  - Non-Violence
  - Weapon Presence
- Data preprocessing includes:
  - Frame extraction
  - Normalization
  - Train / Validation / Test splitting

---

## Project Structure
AI-Violence-Detection-System/
├── ai-model/
│ ├── notebooks/
│ ├── src/
│ ├── weights/
│ └── configs/
├── backend/ # Laravel backend
├── frontend/ # Web dashboard
├── datasets/
├── docs/
└── README.md


---

## Technologies Used
### AI & Machine Learning
- Python
- TensorFlow / PyTorch
- OpenCV
- YOLO
- NumPy, Matplotlib

### Backend
- Laravel (PHP)
- REST APIs
- MySQL

### Frontend
- Vue.js or React
- Chart and visualization libraries

---

## Deployment Strategy
- **Model Training**: Google Colab (GPU acceleration)
- **Inference**: Local server / edge device
- **Backend**: Laravel API server
- **Frontend**: Web-based dashboard

Trained model weights are exported and integrated into the inference pipeline.

---

## Alert Data Format
```json
{
  "camera_id": "CAM_01",
  "district": "Downtown",
  "event_type": "Violence Detected",
  "confidence": 0.92,
  "timestamp": "2026-01-25T18:45:30",
  "snapshot_url": "/storage/incidents/CAM_01_20260125.jpg"
}

```
# Performance Goals

- Real-time inference (low latency)

- High detection accuracy

- Minimal false positives

- Scalable for multi-camera environments

# Project Timeline

- Research and literature review

- Dataset collection and preprocessing

- Model design and training

- Backend and frontend development

- Integration and real-time testing

- Final evaluation and documentation

# Academic References

We are inspired by prior research in:

- Violence detection using CNN + LSTM

- 3D Convolutional Neural Networks

- End-to-end deep learning for video understanding

- Smart surveillance systems

# This project is developed for academic purposes only.