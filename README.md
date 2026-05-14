<div align="center">

# FPGA CNN Fatigue Monitoring

### Hardware-software fatigue monitoring prototype with Python, MediaPipe, Verilog, ModelSim, and Intel Quartus

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange)
![Verilog](https://img.shields.io/badge/Verilog-RTL-red)
![FPGA](https://img.shields.io/badge/FPGA-Cyclone%20IV-purple)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

</div>

---

## Overview

This project is a **hardware-software co-design prototype** for fatigue monitoring.

The software side uses **Python, OpenCV, and MediaPipe FaceMesh** to detect facial landmarks, extract eye and mouth regions, calculate fatigue-related metrics, and generate FPGA-compatible input files.

The FPGA side is designed as a lightweight **CNN-style feature extractor** for small grayscale face regions. It currently focuses on fixed-kernel convolution and is not yet a full trainable CNN accelerator.

---

## Core Idea

```text
Camera Frame
     |
     v
Python + OpenCV
     |
     v
MediaPipe FaceMesh
     |
     v
Eye / Mouth ROI Extraction
     |
     v
32x32 Grayscale ROI
     |
     v
FPGA CNN-Style Feature Extractor
     |
     v
Python Verification and Fatigue Logic
```

The current system extracts three ROIs:

| ROI | Description | FPGA Input Size |
|---|---|---|
| Left Eye | Used for eye closure and blink-related features | 32x32 grayscale |
| Right Eye | Used for eye closure and blink-related features | 32x32 grayscale |
| Mouth | Used for mouth opening and yawn-related features | 32x32 grayscale |

---

## What This Project Does

| Part | Responsibility |
|---|---|
| Python | Camera input, FaceMesh, ROI extraction, EAR, MAR, PERCLOS, logging, golden model |
| FPGA | Fixed-kernel convolution, ReLU, MaxPool, feature map generation |
| Golden Model | Python reference output for checking FPGA simulation results |
| Logs | CSV/XLSX fatigue monitoring output |

---

## What This Project Is Not Yet

This repository is intentionally written as an engineering prototype, not an exaggerated finished product.

It is not yet:

- A full YOLO or object detection system
- A full trainable CNN running completely on FPGA
- A complete real-time deployed hardware product
- A multi-layer CNN accelerator with learned weights
- A medical diagnosis tool

The current FPGA logic is best described as a **proof-of-concept fixed-kernel CNN-style accelerator**.

---

## FPGA Processing Pipeline

The current FPGA-side concept uses a fixed `3x3` convolution kernel. One implemented kernel is a vertical edge-detection kernel:

```text
-1   0   1
-1   0   1
-1   0   1
```

For a `3x3` window:

```text
p00  p01  p02
p10  p11  p12
p20  p21  p22
```

The hardware-friendly equation is:

```text
conv = (p02 + p12 + p22) - (p00 + p10 + p20)
```

This avoids multipliers for this kernel and makes the design suitable for a small FPGA prototype.

```text
32x32 input image
        |
        v
30x30 convolution output
        |
        v
30x30 ReLU output
        |
        v
15x15 MaxPool output
```

---

## System Architecture

```text
Camera / Video Source
        |
        v
Python + OpenCV
        |
        v
MediaPipe FaceMesh
        |
        v
Facial Landmark Detection
        |
        v
ROI Extraction
        |
        +---- left_eye  32x32 grayscale
        +---- right_eye 32x32 grayscale
        +---- mouth     32x32 grayscale
        |
        v
Save FPGA Input Files
        |
        +---- left_eye.hex
        +---- right_eye.hex
        +---- mouth.hex
        |
        v
FPGA CNN-Style Feature Extractor
        |
        v
FPGA Feature Output
        |
        +---- fpga_features_left_eye.txt
        +---- fpga_features_right_eye.txt
        +---- fpga_features_mouth.txt
        |
        v
Python Golden Model Verification
        |
        v
MATCH / MISMATCH
```

In the current software-only run, Python can generate the FPGA input files and compute the golden reference output without real FPGA hardware.

---

## Repository Files

| File | Purpose |
|---|---|
| `main.py` | Main fatigue monitoring application |
| `config.py` | Project settings, thresholds, output paths, camera source |
| `camera.py` | Camera or IP camera input handler |
| `face_landmarks.py` | MediaPipe FaceMesh wrapper |
| `fatigue_metrics.py` | EAR and MAR calculation |
| `fatigue_state.py` | Blink, yawn, PERCLOS, microsleep, and state tracking |
| `fpga_preprocess.py` | Converts frames or ROIs to 32x32 grayscale FPGA input |
| `golden_model.py` | Python reference model for FPGA verification |
| `logger.py` | CSV and Excel logging |
| `.gitignore` | Ignores generated outputs, logs, cache files, and local environment folders |

---

## Generated Files

The Python pipeline may generate these files during runtime:

```text
fpga_input.txt
fpga_input.hex
golden_output.txt
golden_output.hex
fatigue_log.csv
fatigue_log.xlsx
```

These files are not meant to be committed. They are ignored by `.gitignore`.

---

## Requirements

### Python Side

Recommended environment:

```text
Python 3.10 or newer
OpenCV
MediaPipe
NumPy
Pandas
OpenPyXL
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy pandas openpyxl
```

### FPGA Side

```text
Intel Quartus II / Quartus Prime
ModelSim-Altera
Verilog HDL
Cyclone IV FPGA board
```

Target FPGA used in the project plan:

```text
FPGA family: Intel / Altera Cyclone IV
Target device: EP4CE6E22C8
HDL: Verilog
Simulation: ModelSim-Altera
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/TheLouisMahdi/fpga-cnn-fatigue-monitoring.git
cd fpga-cnn-fatigue-monitoring
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe numpy pandas openpyxl
```

### 4. Configure the camera

Open `config.py` and edit `VIDEO_SOURCE`.

For a normal webcam:

```python
VIDEO_SOURCE = 0
```

For an IP camera or DroidCam-style stream:

```python
VIDEO_SOURCE = "http://YOUR_PHONE_IP:8080/video"
```

Example:

```python
VIDEO_SOURCE = "http://10.74.248.143:8080/video"
```

### 5. Run the application

```bash
python main.py
```

Press `Esc` to close the OpenCV window.

---

## Run the Golden Model Only

The golden model is the Python reference version of the FPGA CNN-style pipeline.

```bash
python golden_model.py
```

It creates sample output files:

```text
golden_output.txt
golden_output.hex
```

These outputs can be compared against FPGA or Verilog simulation results.

---

## Verification Flow

```text
Python generates FPGA input
        |
        v
FPGA / Verilog simulation processes the same input
        |
        v
Python golden model processes the same input
        |
        v
Compare FPGA output with golden output
        |
        v
MATCH / MISMATCH
```

Common causes of mismatch:

| Cause | Explanation |
|---|---|
| Kernel mismatch | Python and Verilog use different kernels |
| Addressing mismatch | RAM read order differs from Python flattening order |
| Signed mismatch | Verilog signed values do not match Python integer behavior |
| ReLU mismatch | Negative values are not clipped equally |
| Pooling mismatch | MaxPool window order or stride is different |
| HEX format mismatch | File format is not read consistently |

---

## Current Project Status

| Feature | Status |
|---|---|
| Python camera pipeline | Done |
| MediaPipe FaceMesh integration | Done |
| EAR / MAR calculation | Done |
| Fatigue state tracking | Done |
| CSV / XLSX logging | Done |
| FPGA input generation | Done |
| Python golden model | Done |
| Single fixed-kernel Verilog pipeline | Prototype |
| UART communication | Not completed |
| Real FPGA board test | Not completed |
| Multi-kernel support | Planned |
| Multi-layer CNN support | Planned |
| Learned weight loading | Planned |

---

## Roadmap

- [ ] Clean and finalize RTL folder structure
- [ ] Add Verilog modules and testbenches to the repository
- [ ] Add UART communication between Python and FPGA
- [ ] Add multi-kernel feature extraction
- [ ] Add line-buffer based streaming architecture
- [ ] Add real hardware test results
- [ ] Add screenshots, waveform images, and demo video
- [ ] Add a formal project report

---

## Notes About Accuracy

This project is an educational and engineering prototype.

The fatigue metrics such as EAR, MAR, PERCLOS, blink count, yawn count, and microsleep count are useful for experimentation, but they should not be treated as medical diagnosis.

The FPGA feature extractor is experimental and should be evaluated against the Python golden model before being used as a reliable accelerator.

---

## Author

**Mahdi Ghahremani**

```text
FPGA Design
Verilog RTL
Computer Vision
Embedded Systems
CNN Acceleration
Signal Processing
Hardware / Software Co-Design
```
