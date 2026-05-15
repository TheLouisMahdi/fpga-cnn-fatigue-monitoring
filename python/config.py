# config.py

from pathlib import Path


# ============================================================
# Camera / Video Source
# ============================================================

# IP Webcam / DroidCam examples:
# VIDEO_SOURCE = "http://10.119.67.20:8080/video"
# VIDEO_SOURCE = "http://172.18.20.103:8080/shot.jpg"

# Default webcam:
VIDEO_SOURCE = 0


# ============================================================
# Frame Settings
# ============================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ============================================================
# Fatigue Detection Thresholds
# ============================================================

EAR_CLOSED_THRESHOLD = 0.20
EAR_OPEN_THRESHOLD = 0.23

MAR_YAWN_THRESHOLD = 0.65
YAWN_MIN_DURATION = 0.8

MICROSLEEP_MIN_DURATION = 0.5

PERCLOS_WINDOW_SEC = 60.0
PERCLOS_FATIGUE_THRESHOLD = 0.22


# ============================================================
# FPGA Input Settings
# ============================================================

# FPGA receives 32x32 grayscale ROI images.
FPGA_IMAGE_SIZE = 32


# ============================================================
# Project Paths
# ============================================================
# Expected project structure:
#
# cnn_fpga_project/
# ├── python/
# │   ├── main.py
# │   ├── config.py
# │   └── ...
# └── FPGAp/
#     └── mem/
#
# These paths are portable and GitHub-friendly.
# ============================================================

PYTHON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIR.parent

# Main FPGA project folder.
FPGA_PROJECT_DIR = str(PROJECT_ROOT / "FPGAp")

# ModelSim / Quartus memory folder.
# Python saves left_eye.hex, right_eye.hex, and mouth.hex here.
FPGA_MEM_DIR = str(Path(FPGA_PROJECT_DIR) / "mem")

# Optional local ROI folder for backup/debug.
FPGA_ROI_DIR = str(PYTHON_DIR / "fpga_rois")


# ============================================================
# Logging Settings
# ============================================================

LOG_INTERVAL_SEC = 5.0

OUTPUT_CSV = "fatigue_log.csv"
OUTPUT_XLSX = "fatigue_log.xlsx"


# ============================================================
# UI Settings
# ============================================================

DRAW_UI = True
