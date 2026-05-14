# config.py

# ============================================================
# Camera / Video Source
# ============================================================

#VIDEO_SOURCE = "http://10.119.67.20:8080/video"

# Alternative sources:
# VIDEO_SOURCE = "http://172.18.20.103:8080/shot.jpg"
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

# Main FPGA project folder
FPGA_PROJECT_DIR = r"C:\Users\poimu\Desktop\cnn_fpga_project\FPGAp"

# ModelSim / Quartus memory folder.
# Python should save left_eye.hex, right_eye.hex, and mouth.hex here.
FPGA_MEM_DIR = r"C:\Users\poimu\Desktop\cnn_fpga_project\FPGAp\mem"

# Optional local ROI folder for backup/debug.
FPGA_ROI_DIR = "fpga_rois"


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