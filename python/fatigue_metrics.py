# fatigue_metrics.py

import numpy as np


# ============================================================
# MediaPipe FaceMesh landmark indices
# ============================================================

# Eye landmarks used for EAR
# Order:
# [outer_corner, upper_1, upper_2, inner_corner, lower_2, lower_1]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks used for MAR
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


# ============================================================
# Basic geometry helpers
# ============================================================

def distance(p1, p2):
    """
    Calculate Euclidean distance between two 2D points.

    p1, p2:
        tuple/list/array like (x, y)

    return:
        float distance
    """
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def landmarks_to_pixel_dict(face_landmarks, frame_width, frame_height):
    """
    Convert MediaPipe normalized landmarks to pixel coordinates.

    MediaPipe gives x and y in normalized range:
        x: 0 to 1
        y: 0 to 1

    This function converts them to:
        x pixel
        y pixel

    return:
        dictionary:
            landmarks[index] = (x, y)
    """
    landmarks = {}

    for idx, lm in enumerate(face_landmarks.landmark):
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        landmarks[idx] = (x, y)

    return landmarks


def has_required_landmarks(landmarks, indices):
    """
    Check if all required landmark indices exist.
    """
    for idx in indices:
        if idx not in landmarks:
            return False
    return True


# ============================================================
# EAR / MAR calculations
# ============================================================

def eye_aspect_ratio(points):
    """
    Calculate Eye Aspect Ratio.

    Expected point order:
        p1, p2, p3, p4, p5, p6

    Formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Meaning:
        Smaller EAR usually means eye is more closed.
        Larger EAR usually means eye is more open.
    """
    if len(points) != 6:
        raise ValueError("eye_aspect_ratio needs exactly 6 points")

    p1, p2, p3, p4, p5, p6 = points

    A = distance(p2, p6)
    B = distance(p3, p5)
    C = distance(p1, p4) + 1e-6

    return float((A + B) / (2.0 * C))


def mouth_aspect_ratio(landmarks):
    """
    Calculate Mouth Aspect Ratio.

    Formula:
        MAR = mouth_vertical / mouth_horizontal

    Meaning:
        Larger MAR usually means mouth is more open.
    """
    required = [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]

    if not has_required_landmarks(landmarks, required):
        raise ValueError("Missing mouth landmarks for MAR calculation")

    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical = distance(top, bottom)
    horizontal = distance(left, right) + 1e-6

    return float(vertical / horizontal)


def calculate_ear_mar(face_landmarks, frame_width, frame_height):
    """
    Backward-compatible function.

    This keeps the old main.py working.

    return:
        ear, mar
    """
    metrics = calculate_face_metrics(face_landmarks, frame_width, frame_height)
    return metrics["ear"], metrics["mar"]


def calculate_face_metrics(face_landmarks, frame_width, frame_height):
    """
    Full metric calculation for the current project.

    return:
        {
            "ear": average EAR,
            "left_ear": left eye EAR,
            "right_ear": right eye EAR,
            "mar": mouth aspect ratio,
            "landmarks_px": pixel landmark dictionary
        }

    Why useful for the new FPGA project?
        - ear and mar are used by FatigueState
        - left_ear/right_ear can later be compared with FPGA left/right eye features
        - landmarks_px can be reused for debugging or ROI validation
    """
    landmarks = landmarks_to_pixel_dict(
        face_landmarks=face_landmarks,
        frame_width=frame_width,
        frame_height=frame_height
    )

    if not has_required_landmarks(landmarks, LEFT_EYE):
        raise ValueError("Missing left eye landmarks")

    if not has_required_landmarks(landmarks, RIGHT_EYE):
        raise ValueError("Missing right eye landmarks")

    left_eye_points = [landmarks[i] for i in LEFT_EYE]
    right_eye_points = [landmarks[i] for i in RIGHT_EYE]

    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)

    ear = float((left_ear + right_ear) / 2.0)
    mar = mouth_aspect_ratio(landmarks)

    return {
        "ear": ear,
        "left_ear": left_ear,
        "right_ear": right_ear,
        "mar": mar,
        "landmarks_px": landmarks,
    }


# ============================================================
# Simple standalone test
# ============================================================

if __name__ == "__main__":
    print("fatigue_metrics.py loaded successfully.")
    print("This file needs MediaPipe face_landmarks from main.py to calculate real EAR/MAR.")