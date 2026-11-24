# AI-Based Motorcycle Helmet Detection and Rider Safety Monitoring System Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20Object%20Detection-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

A complete, production-grade Deep Learning traffic surveillance system engineered to automatically detect motorcyclists, classify protective helmet compliance, track violators across traffic surveillance video feeds, display real-time safety analytics, and generate downloadable official compliance audit reports.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Project Directory Structure](#-project-directory-structure)
5. [Quick Start & Installation (Windows Laptop)](#-quick-start--installation-windows-laptop)
6. [Feature Walkthrough & Guide](#-feature-walkthrough--guide)
7. [Mathematical Formulations](#-mathematical-formulations)
8. [Academic Viva Voce Q&A](#-academic-viva-voce-qa)
9. [Author & Academic Submission](#-author--academic-submission)

---

## 🎯 Project Overview

Two-wheeler road accidents represent over 37% of road fatalities globally. The majority of catastrophic injuries result from head trauma that could be mitigated by wearing approved safety helmets. Manual physical enforcement checkpoints:
- Cause heavy traffic congestion.
- Pose safety risks to on-duty police personnel.
- Cannot scale to 24/7 monitoring across multi-lane urban corridors.

This project delivers an automated, intelligent surveillance solution combining **YOLOv8 deep learning object detection**, **two-stage spatial association**, **multi-object tracking**, and an interactive **Streamlit dashboard**.

---

## 🚀 Key Features

| # | Feature | Capability |
|---|---|---|
| 1 | **Executive Surveillance Dashboard** | Real-time KPI scorecard: Total Scanned, Compliant Helmets, Violations, Compliance Rate (%), and Safety Risk Zone status. |
| 2 | **Image Detection Engine** | High-precision single/batch image detection with color-coded bounding boxes (Lime Green for Helmet, Bright Red for Violation). Includes zoomable comparison, head ROI crop gallery, and confidence scores. |
| 3 | **Traffic Video Surveillance** | Frame-by-frame inference with live telemetry HUD, multi-object IoU tracking to eliminate double-counting, temporal violation timeline chart, and downloadable annotated MP4 video. |
| 4 | **Real-Time Webcam Detection** | Continuous webcam feed and instant snapshot mode with automated snapshot capture for violators stored in `data/violations/`. |
| 5 | **Interactive Analytics & Charts** | Plotly charts: Compliance Donut, Category Breakdown Bar Chart, Safety Index Gauge, Confidence Distribution Histogram, and Source Distribution Pie Chart. |
| 6 | **Downloadable Audit Reports** | Executive PDF Compliance Audit Reports (with ReportLab tables, KPI metrics, and evaluator sign-off blocks) + CSV Telemetry Incident Logs. |
| 7 | **Built-in Sample Datasets** | Bundled high-resolution synthetic traffic images and traffic surveillance video for immediate testing without external files. |
| 8 | **Academic Presentation Guide** | Integrated college viva voce tab with architecture flowcharts, mathematical formulas, and the top 10 viva evaluation questions with answers. |

---

## 🏗️ System Architecture

```
+-------------------------------------------------------------------------+
|                 Input Stream (Images / Video / Webcam)                  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  YOLOv8 Deep Feature Extraction                         |
|   - Backbone: Modified CSPDarknet53 with C2f Cross-Stage Feature Fusion |
|   - Neck: PANet (Path Aggregation Network) Multi-Scale Pyramid           |
|   - Head: Anchor-Free Decoupled Classification & Regression Heads       |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|              Spatial Association & Head ROI Localization                |
|   - Motorcyclist Isolation: Person overlapping Motorcycle Bounding Box  |
|   - Head ROI Extraction: Upper 25% of Bounding Box Height               |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                   Deep Feature Helmet Classification                    |
|   - Curvature and Edge Smoothness Analysis                              |
|   - Skin / Hair Chrominance Signature Filtering                         |
|   - Helmet Visor & Shell Reflectance Profiling                          |
+-------------------------------------------------------------------------+
                                     |
                  +------------------+------------------+
                  |                                     |
                  v                                     v
       [ Class: Helmet ]                    [ Class: Without Helmet ]
      Color: Green (#32CD32)                  Color: Red (#E62828)
       Status: COMPLIANT                     Status: SAFETY VIOLATION
                  |                                     |
                  +------------------+------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|              Multi-Object IoU Tracker & Persistent IDs                  |
|   - Tracks riders across consecutive frames to avoid double-counting    |
|   - Automatically archives violation crops to data/violations/          |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|        Streamlit Web Dashboard, Telemetry HUD & PDF Audit Reports       |
+-------------------------------------------------------------------------+
```

---

## 📁 Project Directory Structure

```
AI-DL-project/
│
├── app.py                      # Main Streamlit web application & user interface
├── requirements.txt            # Pinned project dependencies
├── run.bat                     # 1-Click Windows execution script
├── setup.bat                   # 1-Click Windows environment setup script
├── run_app.py                  # Direct Python launcher script
├── test_system.py              # Verification & diagnostic test script
├── README.md                   # Complete project documentation & guide
│
├── modules/                    # Core project modular architecture
│   ├── __init__.py
│   ├── config.py               # Paths, classes, color maps, thresholds
│   ├── detector.py             # YOLOv8 Deep Learning & Head ROI detection engine
│   ├── tracker.py              # Centroid & IoU Multi-Object Tracker (persistent IDs)
│   ├── analytics.py            # Plotly interactive charts & telemetry manager
│   ├── report_generator.py     # Executive PDF & CSV audit report generator
│   └── utils.py                # Video processor, HUD telemetry, synthetic media
│
├── models/                     # Deep learning model weights directory
│   ├── download_weights.py     # Automated weights downloader & verifier
│   └── yolov8n.pt              # YOLOv8 neural network weights
│
├── data/                       # Application data & runtime storage
│   ├── samples/                # Built-in synthetic sample traffic images & video
│   ├── violations/             # Automatic snapshots vault for non-compliant riders
│   ├── reports/                # Exported PDF and CSV audit reports
│   ├── outputs/                # Annotated output media
│   └── uploads/                # Temporary uploaded media files
│
└── logs/                       # Application event logs
    └── detection_history.csv   # Persistent detection telemetry log
```

---

## 💻 Quick Start & Installation (Windows Laptop)

### Method 1: One-Click Execution (Recommended)
Simply **double-click** the `run.bat` file in the project folder!
1. It automatically detects your Python environment.
2. Creates the isolated virtual environment (`.venv`).
3. Installs all required packages.
4. Generates test samples and verifies model weights.
5. Launches the web app at `http://localhost:8501` in your browser.

---

### Method 2: Manual Terminal Execution

1. Open PowerShell or Command Prompt in the project folder:
   ```powershell
   cd c:\Users\krish\Downloads\AI-DL-project
   ```

2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Download model weights:
   ```powershell
   python models\download_weights.py
   ```

5. Run the Streamlit application:
   ```powershell
   streamlit run app.py
   ```

6. Open your browser and navigate to `http://localhost:8501`.

---

## 🎨 Feature Walkthrough & Demonstration Guide

### 1. Executive Dashboard (Tab 1)
- View high-level safety statistics: Total Records, Compliant Riders, Violations, Compliance %, and Safety Risk Zone.
- Inspect the interactive **Compliance Donut** and **Rider Safety Index Gauge**.
- Review the **Safety Violations Vault** showing recent captured snapshots.

### 2. Image Detection (Tab 2)
- Select from 3 pre-loaded samples or upload custom traffic images.
- Adjust **Confidence** and **IoU** thresholds in real-time from the sidebar.
- Examine side-by-side detection comparisons, color-coded bounding boxes, and extracted head ROI crops.
- Download annotated images and inspection logs.

### 3. Traffic Video Inspection (Tab 3)
- Process surveillance video feeds frame-by-frame.
- Real-time video preview with telemetry HUD overlay: Active FPS, Frame counter, and Live counts.
- Persistent Track IDs (`ID:1`, `ID:2`) via IoU Multi-Object Tracker.
- Temporal violation line chart and downloadable processed MP4 video.

### 4. Real-Time Webcam (Tab 4)
- **Snapshot Mode:** Snap photos from your laptop webcam with instant AI evaluation.
- **Continuous Mode:** Real-time OpenCV streaming feed with live FPS overlay.
- Automatic snapshot saving of flagged violators to `data/violations/`.

### 5. Analytics & Charts (Tab 5)
- Interactive Plotly charts: Category Bar Chart, Donut Chart, Confidence Histogram, and Source Pie Chart.
- Filter historical telemetry by input source (Image, Video, Webcam) or class category.

### 6. Audit Reports & Exports (Tab 6)
- **Executive PDF Report:** Generates a publication-grade PDF with compliance scorecards, incident details, and evaluator sign-off blocks.
- **CSV Data Log:** Exports full telemetry logs for Excel/spreadsheet analysis.

### 7. Project Guide & Viva Q&A (Tab 7)
- Built-in documentation covering problem statement, system architecture, mathematical formulas, and the top 10 college viva questions with answers.

---

## 📐 Mathematical Formulations

### 1. Intersection over Union (IoU)
$$\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$

### 2. Precision & Recall
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### 3. F1-Score
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 4. Mean Average Precision (mAP@0.50)
$$\text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \text{AP}_i, \quad \text{AP} = \int_{0}^{1} p(r) dr$$

---

## 🎓 Academic Viva Voce Q&A

**Q1: What is the main innovation of your project?**  
*Answer:* Rather than relying on a simple single-class detector, this system uses a multi-stage architecture that associates riders with two-wheelers, isolates head regions of interest, and applies multi-object tracking to prevent double-counting across video frames.

**Q2: Why use YOLOv8 instead of Faster R-CNN?**  
*Answer:* Faster R-CNN requires a two-stage Region Proposal Network (RPN) which runs at only ~5-10 FPS on CPU. YOLOv8 is an anchor-free single-stage detector delivering 45+ FPS real-time throughput while maintaining high accuracy.

**Q3: How does the tracking algorithm prevent duplicate violation counts?**  
*Answer:* We implemented an IoU & Centroid Multi-Object Tracker (`modules/tracker.py`). Each rider is assigned a unique Track ID. A set of seen violation IDs ensures each non-compliant rider is recorded only once in the overall traffic compliance statistics.

---

## 📄 Academic Submission Details

- **Project Title:** AI-Based Motorcycle Helmet Detection and Rider Safety Monitoring System Using Deep Learning
- **Frameworks:** Python, YOLOv8, PyTorch, OpenCV, Streamlit, Plotly, ReportLab
- **Hardware Compatibility:** CPU-Optimized (Intel/AMD) & CUDA GPU Accelerated
- **Target OS:** Windows 10 / 11, Linux, macOS
