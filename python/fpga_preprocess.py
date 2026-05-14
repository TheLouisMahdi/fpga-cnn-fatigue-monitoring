# fpga_preprocess.py

import os
import cv2
import numpy as np

from config import FPGA_IMAGE_SIZE

try:
    from config import FPGA_MEM_DIR
except ImportError:
    FPGA_MEM_DIR = "fpga_rois"

try:
    from config import FPGA_ROI_DIR
except ImportError:
    FPGA_ROI_DIR = "fpga_rois"


# ============================================================
# MediaPipe FaceMesh landmark indices for FPGA ROIs
# ============================================================
# These landmarks are used only to find crop boxes.
# Final FPGA input is always 32x32 grayscale uint8.

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 291, 13, 14, 78, 308]


# ============================================================
# ROI names used by FPGA and ModelSim
# ============================================================

ROI_LEFT_EYE = "left_eye"
ROI_RIGHT_EYE = "right_eye"
ROI_MOUTH = "mouth"

REQUIRED_ROIS = [
    ROI_LEFT_EYE,
    ROI_RIGHT_EYE,
    ROI_MOUTH,
]


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def validate_fpga_image(img_32x32, name="roi"):
    """
    Validate that image is suitable for FPGA input.

    FPGA expects:
        shape: 32x32
        dtype: uint8
        range: 0 to 255
    """
    if img_32x32 is None:
        raise ValueError(f"{name} is None")

    if img_32x32.shape != (FPGA_IMAGE_SIZE, FPGA_IMAGE_SIZE):
        raise ValueError(
            f"{name} has invalid shape {img_32x32.shape}, "
            f"expected {(FPGA_IMAGE_SIZE, FPGA_IMAGE_SIZE)}"
        )

    if img_32x32.dtype != np.uint8:
        raise ValueError(f"{name} dtype is {img_32x32.dtype}, expected uint8")

    return True


def flatten_for_uart(img_32x32):
    """
    Convert 32x32 image to 1D uint8 array.

    Important:
        numpy flatten uses row-major order by default.

    Address mapping in FPGA:
        addr = row * 32 + col

    So:
        flat[0]    = row 0, col 0
        flat[1]    = row 0, col 1
        flat[32]   = row 1, col 0
        flat[1023] = row 31, col 31
    """
    validate_fpga_image(img_32x32)
    return img_32x32.flatten().astype(np.uint8)


# ============================================================
# ROI extraction
# ============================================================

def crop_roi_from_landmarks(frame, face_landmarks, indices, padding=20):
    """
    Crop one ROI from frame using selected FaceMesh landmarks.

    Input:
        frame:
            BGR OpenCV frame

        face_landmarks:
            MediaPipe FaceMesh landmarks

        indices:
            list of landmark indices

        padding:
            number of pixels added around bounding box

    Output:
        cropped BGR ROI or None
    """
    if frame is None or face_landmarks is None:
        return None

    h, w = frame.shape[:2]

    if h <= 0 or w <= 0:
        return None

    xs = []
    ys = []

    for idx in indices:
        if idx >= len(face_landmarks.landmark):
            return None

        lm = face_landmarks.landmark[idx]

        x = int(lm.x * w)
        y = int(lm.y * h)

        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    x1 = max(min(xs) - padding, 0)
    y1 = max(min(ys) - padding, 0)
    x2 = min(max(xs) + padding, w)
    y2 = min(max(ys) + padding, h)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return roi


def preprocess_roi_for_fpga(roi):
    """
    Convert one BGR ROI to FPGA-ready image.

    Steps:
        1. BGR to grayscale
        2. Resize to 32x32
        3. Convert to uint8

    Output:
        32x32 uint8 grayscale image
    """
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        (FPGA_IMAGE_SIZE, FPGA_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    fpga_img = resized.astype(np.uint8)

    validate_fpga_image(fpga_img)

    return fpga_img


def extract_fpga_rois(frame, face_landmarks):
    """
    Extract all FPGA ROIs from current frame.

    Output dictionary:
        {
            "left_eye":  32x32 uint8 image,
            "right_eye": 32x32 uint8 image,
            "mouth":     32x32 uint8 image
        }
    """
    rois = {}

    left_eye_raw = crop_roi_from_landmarks(
        frame=frame,
        face_landmarks=face_landmarks,
        indices=LEFT_EYE_IDX,
        padding=18
    )

    right_eye_raw = crop_roi_from_landmarks(
        frame=frame,
        face_landmarks=face_landmarks,
        indices=RIGHT_EYE_IDX,
        padding=18
    )

    mouth_raw = crop_roi_from_landmarks(
        frame=frame,
        face_landmarks=face_landmarks,
        indices=MOUTH_IDX,
        padding=25
    )

    left_eye = preprocess_roi_for_fpga(left_eye_raw)
    right_eye = preprocess_roi_for_fpga(right_eye_raw)
    mouth = preprocess_roi_for_fpga(mouth_raw)

    if left_eye is not None:
        rois[ROI_LEFT_EYE] = left_eye

    if right_eye is not None:
        rois[ROI_RIGHT_EYE] = right_eye

    if mouth is not None:
        rois[ROI_MOUTH] = mouth

    return rois


# ============================================================
# Save functions
# ============================================================

def save_array_txt(img_32x32, path):
    """
    Save FPGA input image as decimal TXT.

    Format:
        one decimal pixel per line

    Example:
        160
        161
        159

    This is useful for debugging.
    """
    validate_fpga_image(img_32x32, name=os.path.basename(path))

    flat = flatten_for_uart(img_32x32)

    with open(path, "w", encoding="utf-8") as f:
        for value in flat:
            f.write(f"{int(value)}\n")


def save_array_hex(img_32x32, path):
    """
    Save FPGA input image as HEX.

    Format:
        one 8-bit hex pixel per line

    Example:
        A0
        A1
        9F

    This is the format used by Verilog $readmemh.
    """
    validate_fpga_image(img_32x32, name=os.path.basename(path))

    flat = flatten_for_uart(img_32x32)

    with open(path, "w", encoding="utf-8") as f:
        for value in flat:
            f.write(f"{int(value):02X}\n")


def save_fpga_rois(rois, output_dir=None, save_txt=True, save_hex=True):
    """
    Save FPGA ROI images to disk.

    Default output:
        FPGA_MEM_DIR if available
        otherwise fpga_rois

    Files:
        left_eye.txt
        left_eye.hex
        right_eye.txt
        right_eye.hex
        mouth.txt
        mouth.hex

    Return:
        dictionary of saved file paths
    """
    if output_dir is None:
        output_dir = FPGA_MEM_DIR

    ensure_dir(output_dir)

    saved_paths = {}

    for name in REQUIRED_ROIS:
        if name not in rois:
            continue

        img = rois[name]
        validate_fpga_image(img, name=name)

        saved_paths[name] = {}

        if save_txt:
            txt_path = os.path.join(output_dir, f"{name}.txt")
            save_array_txt(img, txt_path)
            saved_paths[name]["txt"] = txt_path

        if save_hex:
            hex_path = os.path.join(output_dir, f"{name}.hex")
            save_array_hex(img, hex_path)
            saved_paths[name]["hex"] = hex_path

    return saved_paths


def save_fpga_rois_backup(rois, output_dir=None):
    """
    Optional backup save into local fpga_rois folder.

    This is useful if you want a copy separate from ModelSim mem folder.
    """
    if output_dir is None:
        output_dir = FPGA_ROI_DIR

    return save_fpga_rois(
        rois=rois,
        output_dir=output_dir,
        save_txt=True,
        save_hex=True
    )


# ============================================================
# Legacy whole-frame path
# ============================================================

def preprocess_for_fpga(frame):
    """
    Legacy path:
        Convert full frame to 32x32 grayscale uint8.

    This is kept for backward compatibility with older tests.

    New FPGA path should use:
        extract_fpga_rois()
        save_fpga_rois()
    """
    if frame is None or frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        (FPGA_IMAGE_SIZE, FPGA_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    fpga_input = resized.astype(np.uint8)

    validate_fpga_image(fpga_input, name="whole_frame_fpga_input")

    return fpga_input


def save_fpga_input_txt(img_32x32, path="fpga_input.txt"):
    """
    Legacy save function for old whole-frame tests.
    """
    save_array_txt(img_32x32, path)


def save_fpga_input_hex(img_32x32, path="fpga_input.hex"):
    """
    Legacy save function for old whole-frame tests.
    """
    save_array_hex(img_32x32, path)


# ============================================================
# Standalone check
# ============================================================

if __name__ == "__main__":
    print("fpga_preprocess.py loaded successfully.")
    print(f"FPGA_IMAGE_SIZE = {FPGA_IMAGE_SIZE}")
    print(f"Default FPGA_MEM_DIR = {FPGA_MEM_DIR}")