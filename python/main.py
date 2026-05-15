# main.py

import cv2
import time
import os

from camera import Camera
from face_landmarks import FaceLandmarks
from fatigue_metrics import calculate_face_metrics
from fatigue_state import FatigueState
from logger import DataLogger

from fpga_preprocess import (
    extract_fpga_rois,
    save_fpga_rois,
    save_fpga_rois_backup
)

from golden_model import (
    run_fixed_kernel_feature_extractor,
    save_features_txt
)

from config import (
    VIDEO_SOURCE,
    LOG_INTERVAL_SEC,
    FPGA_MEM_DIR,
    FPGA_ROI_DIR,
    DRAW_UI
)


# ============================================================
# Required FPGA ROI names
# ============================================================

REQUIRED_ROI_SET = {"left_eye", "right_eye", "mouth"}


# ============================================================
# Helper functions
# ============================================================

def draw_text(frame, text, x, y, color=(255, 255, 0), scale=0.75, thickness=2):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def generate_python_golden_features(rois, output_dir=FPGA_MEM_DIR):
    """
    Generate Python golden features for current FPGA ROIs.

    Input:
        rois:
            {
                "left_eye":  32x32 uint8,
                "right_eye": 32x32 uint8,
                "mouth":     32x32 uint8
            }

    Output files:
        python_features_left_eye.txt
        python_features_right_eye.txt
        python_features_mouth.txt

    Return:
        {
            "left_eye":  [18 features],
            "right_eye": [18 features],
            "mouth":     [18 features]
        }
    """

    os.makedirs(output_dir, exist_ok=True)

    features = {}

    for roi_name, img in rois.items():
        result = run_fixed_kernel_feature_extractor(img)
        feature_vector = result["features"]

        features[roi_name] = feature_vector

        output_path = os.path.join(output_dir, f"python_features_{roi_name}.txt")
        save_features_txt(feature_vector, output_path)

    return features


def summarize_feature_vector(features):
    """
    Small UI/debug summary from 18-feature vector.
    This does not replace the full feature file.
    """

    if not features:
        return "none"

    try:
        vertical_sum = features[0]
        horizontal_sum = features[3]
        sharpen_sum = features[12]
        center_sum = features[15]

        return (
            f"V:{vertical_sum} "
            f"H:{horizontal_sum} "
            f"S:{sharpen_sum} "
            f"C:{center_sum}"
        )
    except Exception:
        return "invalid"


# ============================================================
# Initialize modules
# ============================================================

cam = Camera(VIDEO_SOURCE)
face = FaceLandmarks()
state = FatigueState()
logger = DataLogger()

last_log_time = time.time()

# Keep last generated features for UI display.
last_python_features = None


try:
    while True:
        # ========================================================
        # Read frame
        # ========================================================

        frame = cam.read()

        if frame is None:
            continue

        raw_frame = frame.copy()
        h, w = frame.shape[:2]

        # ========================================================
        # FaceMesh processing
        # ========================================================

        result = face.process(frame)

        if DRAW_UI:
            frame = face.draw(frame, result)

        # Default values
        rois = {}
        python_features = None
        saved_paths = {}

        # ========================================================
        # Face detected
        # ========================================================

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]

            # ====================================================
            # EAR / MAR / landmark-based metrics
            # ====================================================

            metrics = calculate_face_metrics(face_landmarks, w, h)

            ear = metrics["ear"]
            mar = metrics["mar"]
            left_ear = metrics["left_ear"]
            right_ear = metrics["right_ear"]

            # ====================================================
            # FPGA ROI extraction
            # ====================================================
            # Output ROIs:
            #   left_eye  -> 32x32 uint8
            #   right_eye -> 32x32 uint8
            #   mouth     -> 32x32 uint8
            # ====================================================

            rois = extract_fpga_rois(raw_frame, face_landmarks)

            now = time.time()
            all_rois_ready = REQUIRED_ROI_SET.issubset(rois.keys())

            # ====================================================
            # Fatigue state update
            # ====================================================
            # Note:
            #   python_features is generated only periodically.
            #   Between save intervals, it stays None.
            #   FatigueState supports fpga_features=None safely.
            # ====================================================

            info = state.update(
                ear,
                mar,
                fpga_features=python_features
            )

            # ====================================================
            # Periodic save + logging
            # ====================================================
            # Heavy disk writes are done only every LOG_INTERVAL_SEC.
            # This avoids writing HEX/TXT files on every camera frame.
            # ====================================================

            if now - last_log_time >= LOG_INTERVAL_SEC:
                if all_rois_ready:
                    saved_paths = save_fpga_rois(
                        rois,
                        output_dir=FPGA_MEM_DIR,
                        save_txt=True,
                        save_hex=True
                    )

                    save_fpga_rois_backup(
                        rois,
                        output_dir=FPGA_ROI_DIR
                    )

                    python_features = generate_python_golden_features(
                        rois,
                        output_dir=FPGA_MEM_DIR
                    )

                    last_python_features = python_features
                else:
                    python_features = None
                    last_python_features = None

                # Re-update state with generated features for this log row.
                # This keeps fpga_available accurate in the saved log.
                info = state.update(
                    ear,
                    mar,
                    fpga_features=python_features
                )

                logger.add(
                    ear,
                    mar,
                    info,
                    fpga_features=python_features
                )

                logger.save_all()

                print("Saved FPGA ROI inputs to:", FPGA_MEM_DIR)
                print("Saved ROI names:", list(rois.keys()))
                print("All ROIs ready:", all_rois_ready)
                print("Saved paths:", saved_paths)

                if python_features:
                    print("Saved Python golden features:", list(python_features.keys()))
                else:
                    print("Python golden features: not generated")

                print("EAR:", round(ear, 4), "MAR:", round(mar, 4))
                print("State:", info["fatigue_state"])
                print("Attention:", info["attention_score"])
                print("-" * 60)

                last_log_time = now

            # ====================================================
            # Draw UI
            # ====================================================

            if DRAW_UI:
                draw_text(frame, f"EAR: {ear:.3f}", 20, 35, color=(0, 255, 255), scale=0.6)
                draw_text(frame, f"L_EAR: {left_ear:.3f}", 20, 60, color=(0, 255, 255), scale=0.6)
                draw_text(frame, f"R_EAR: {right_ear:.3f}", 20, 85, color=(0, 255, 255), scale=0.6)
                draw_text(frame, f"MAR: {mar:.3f}", 20, 110, color=(0, 255, 255), scale=0.6)

                draw_text(frame, f"PERCLOS: {info['perclos']:.2f}", 20, 145, scale=0.6)
                draw_text(frame, f"Closed: {info['closed']}", 20, 170, scale=0.6)
                draw_text(frame, f"CloseDur: {info['current_closure_duration']:.2f}s", 20, 195, scale=0.6)

                draw_text(frame, f"Blink: {info['blink_count']}", 20, 225, scale=0.6)
                draw_text(frame, f"LongClose: {info['long_closure_count']}", 20, 250, scale=0.6)
                draw_text(frame, f"Yawn: {info['yawn_count']}", 20, 275, scale=0.6)
                draw_text(frame, f"Microsleep: {info['microsleep_count']}", 20, 300, scale=0.6)

                draw_text(
                    frame,
                    f"State: {info['fatigue_state']}",
                    20,
                    335,
                    color=(0, 0, 255),
                    scale=0.7
                )

                draw_text(
                    frame,
                    f"Attention: {info['attention_score']}",
                    20,
                    365,
                    color=(0, 0, 255),
                    scale=0.7
                )

                roi_names_text = ",".join(rois.keys()) if rois else "none"

                draw_text(
                    frame,
                    f"ROIs: {roi_names_text}",
                    20,
                    400,
                    color=(0, 255, 0),
                    scale=0.55,
                    thickness=1
                )

                draw_text(
                    frame,
                    f"All ROIs ready: {all_rois_ready}",
                    20,
                    425,
                    color=(0, 255, 0),
                    scale=0.55,
                    thickness=1
                )

                # Show last saved feature summary, not per-frame generated features.
                if last_python_features:
                    left_summary = summarize_feature_vector(
                        last_python_features.get("left_eye")
                    )
                    right_summary = summarize_feature_vector(
                        last_python_features.get("right_eye")
                    )
                    mouth_summary = summarize_feature_vector(
                        last_python_features.get("mouth")
                    )

                    draw_text(
                        frame,
                        f"L FPGA: {left_summary}",
                        330,
                        35,
                        color=(0, 255, 0),
                        scale=0.45,
                        thickness=1
                    )

                    draw_text(
                        frame,
                        f"R FPGA: {right_summary}",
                        330,
                        60,
                        color=(0, 255, 0),
                        scale=0.45,
                        thickness=1
                    )

                    draw_text(
                        frame,
                        f"M FPGA: {mouth_summary}",
                        330,
                        85,
                        color=(0, 255, 0),
                        scale=0.45,
                        thickness=1
                    )

        # ========================================================
        # No face detected
        # ========================================================

        else:
            if DRAW_UI:
                draw_text(
                    frame,
                    "No face detected",
                    20,
                    40,
                    color=(0, 0, 255),
                    scale=0.9
                )

        # ========================================================
        # Show frame
        # ========================================================

        if DRAW_UI:
            cv2.imshow("Fatigue Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break


finally:
    logger.save_all()

    cam.release()

    try:
        face.close()
    except AttributeError:
        pass

    cv2.destroyAllWindows()
