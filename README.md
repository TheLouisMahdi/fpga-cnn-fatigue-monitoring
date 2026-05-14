# FPGA CNN Fatigue Monitoring

FPGA-accelerated CNN feature extraction system for fatigue monitoring using Python, MediaPipe FaceMesh, Verilog, ModelSim and Intel Quartus.

---

## 1. Project Overview

This project implements a hardware-software fatigue monitoring system.

The software side uses Python and MediaPipe FaceMesh to detect facial landmarks, extract important face regions, calculate fatigue-related metrics, and generate FPGA input files.

The FPGA side receives three small grayscale images called ROIs:

1. Left eye
2. Right eye
3. Mouth

Each ROI is converted to a 32x32 grayscale image and saved as a HEX file. The FPGA reads these files, applies fixed CNN-style kernels, performs ReLU, clipping, pooling, and feature summarization.

The final FPGA output is a compact feature vector for each ROI.

---

## 2. Main Goal

The goal is to build a lightweight FPGA-based CNN feature extractor that can support fatigue detection.

Instead of running a full CNN model on the FPGA, this project uses fixed convolution kernels to extract useful visual features from eye and mouth regions.

The final fatigue decision is still handled in Python using:

- EAR
- MAR
- PERCLOS
- Blink count
- Yawn count
- Microsleep count
- FPGA extracted features

---

## 3. System Architecture

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
FPGA CNN Feature Extractor
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