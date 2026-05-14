# FPGA CNN Fatigue Monitoring

FPGA-accelerated CNN-style feature extraction system for fatigue monitoring using Python, OpenCV, MediaPipe FaceMesh, Verilog, ModelSim, and Intel Quartus.

This project is a hardware-software co-design prototype. Python handles the camera, facial landmark detection, ROI extraction, fatigue metrics, logging, and verification. The FPGA side is intended to accelerate lightweight CNN-style feature extraction on small face regions.

---

## 1. Project Overview

The project detects fatigue-related visual cues from a camera stream and prepares small grayscale face regions for FPGA processing.

The software side extracts three main ROIs:

1. Left eye
2. Right eye
3. Mouth

Each ROI is converted into a `32x32` grayscale image. These images can be saved as `.hex` files and used as FPGA memory inputs.

The FPGA side is designed as a lightweight CNN-style feature extractor. It currently focuses on fixed kernel-based convolution, not a full trainable deep neural network. The main implemented idea is:

```text
32x32 grayscale ROI
        |
        v
3x3 fixed convolution kernel
        |
        v
ReLU
        |
        v
2x2 MaxPool
        |
        v
Feature map / feature summary
```

The final fatigue decision is handled in Python using classical fatigue metrics and optional FPGA-generated visual features.

---

## 2. What This Project Is and Is Not

### This project is:

- A Python-based fatigue monitoring pipeline
- A MediaPipe FaceMesh-based landmark and ROI extraction system
- A Verilog RTL prototype for CNN-style feature extraction
- A hardware-software co-design experiment
- A step toward FPGA acceleration for computer vision preprocessing

### This project is not yet:

- A full YOLO or object detection system
- A full trainable CNN running completely on FPGA
- A complete real-time deployed hardware product
- A multi-layer CNN accelerator with learned weights

The current FPGA logic is best described as a proof-of-concept fixed-kernel CNN-style accelerator.

---

## 3. Main Goal

The goal is to build a lightweight FPGA-assisted fatigue monitoring system.

Python performs high-level tasks:

```text
Camera input
Face landmark detection
Eye and mouth ROI extraction
EAR calculation
MAR calculation
PERCLOS calculation
Blink / yawn / microsleep tracking
Logging
Golden model verification
```

FPGA performs low-level acceleration tasks:

```text
Input image buffering
3x3 convolution with fixed kernel
ReLU activation
2x2 max pooling
Feature map generation
Feature summarization
```

The design philosophy is:

```text
Python = system control and decision logic
FPGA   = low-level image feature extraction accelerator
```

---

## 4. System Architecture

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

## 5. Current FPGA Processing Pipeline

The current FPGA-side concept uses a fixed 3x3 convolution kernel. One implemented kernel is a vertical edge-detection kernel:

```text
-1   0   1
-1   0   1
-1   0   1
```

For a 3x3 window:

```text
p00  p01  p02
p10  p11  p12
p20  p21  p22
```

The simplified hardware equation is:

```text
conv = (p02 + p12 + p22) - (p00 + p10 + p20)
```

This is efficient for FPGA because it avoids multipliers for this specific kernel.

The basic data flow is:

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

Current limitation: this is a single fixed-kernel feature extraction design. Multi-kernel and multi-layer CNN support are future improvements.

---

## 6. Python Files

| File | Purpose |
|---|---|
| `main.py` | Main fatigue monitoring application |
| `config.py` | Project settings, thresholds, paths, camera source |
| `camera.py` | Camera or IP camera input handler |
| `face_landmarks.py` | MediaPipe FaceMesh wrapper |
| `fatigue_metrics.py` | EAR and MAR calculation |
| `fatigue_state.py` | Blink, yawn, PERCLOS, microsleep, and state tracking |
| `fpga_preprocess.py` | Converts camera frames or ROIs to 32x32 grayscale FPGA input |
| `golden_model.py` | Python reference model for CNN-style FPGA verification |
| `logger.py` | CSV and Excel logging |

---

## 7. Expected Generated Files

When the Python pipeline runs, it may generate files like:

```text
fpga_input.txt
fpga_input.hex
golden_output.txt
golden_output.hex
fatigue_log.csv
fatigue_log.xlsx
```

These files are generated at runtime and are intentionally ignored by Git using `.gitignore`.

---

## 8. Requirements

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

For FPGA development:

```text
Intel Quartus II / Quartus Prime
ModelSim-Altera
Cyclone IV FPGA board
Verilog HDL
```

Target FPGA used in the project plan:

```text
FPGA family: Intel / Altera Cyclone IV
Target device: EP4CE6E22C8
HDL: Verilog
Simulation: ModelSim-Altera
```

---

## 9. How to Run the Python Side

### Step 1: Clone the repository

```bash
git clone https://github.com/TheLouisMahdi/fpga-cnn-fatigue-monitoring.git
cd fpga-cnn-fatigue-monitoring
```

### Step 2: Create a virtual environment

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

### Step 3: Install dependencies

```bash
pip install opencv-python mediapipe numpy pandas openpyxl
```

### Step 4: Configure the camera source

Open `config.py` and set `VIDEO_SOURCE`.

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

### Step 5: Run the main program

```bash
python main.py
```

Press `Esc` to close the OpenCV window.

---

## 10. How to Run the Golden Model Only

The golden model is the Python reference version of the FPGA CNN-style pipeline.

Run:

```bash
python golden_model.py
```

This creates sample golden output files:

```text
golden_output.txt
golden_output.hex
```

These files can be used to compare Python output against FPGA simulation output.

---

## 11. How the Verification Works

The verification idea is:

```text
Python generates FPGA input
        |
        v
FPGA or Verilog simulation processes the same input
        |
        v
Python golden model processes the same input
        |
        v
Compare FPGA output with Python golden output
```

Expected result:

```text
MATCH
```

If the results do not match, possible causes include:

- Different kernel mode between Python and Verilog
- Wrong memory address order
- Signed / unsigned mismatch
- ReLU or clipping difference
- MaxPool indexing mismatch
- HEX file format mismatch

---

## 12. Current Project Status

Completed or partially completed:

- Python camera pipeline
- MediaPipe FaceMesh integration
- EAR and MAR calculation
- Fatigue state tracking
- Logging system
- FPGA input generation
- Python golden model
- Verilog proof-of-concept CNN-style pipeline
- Single fixed-kernel feature extraction simulation

Not fully completed yet:

- Full UART communication between Python and FPGA
- Real hardware test on Cyclone IV board
- Multi-kernel support
- Multi-layer CNN support
- Learned weight loading
- End-to-end real-time FPGA deployment

---

## 13. Roadmap

Planned improvements:

1. Clean and finalize RTL folder structure
2. Add Verilog modules and testbenches to the repository
3. Add UART-based communication between Python and FPGA
4. Add multi-kernel feature extraction
5. Add line-buffer based streaming architecture
6. Add real hardware test results
7. Add screenshots, waveform images, and demo video
8. Add a formal project report

---

## 14. Notes About Accuracy

This project should be understood as an educational and engineering prototype.

The Python fatigue metrics such as EAR, MAR, PERCLOS, blink count, yawn count, and microsleep count are useful for experimentation, but they are not medical diagnosis tools.

The FPGA feature extractor is currently experimental and should be evaluated against the Python golden model before being used as a reliable accelerator.

---

## 15. Author

**Mahdi Ghahremani**

Focus areas:

```text
FPGA Design
Verilog RTL
Computer Vision
Embedded Systems
CNN Acceleration
Signal Processing
Hardware / Software Co-Design
```
