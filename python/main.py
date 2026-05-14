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

            # Save ROI inputs directly into FPGA/ModelSim mem folder.
            # Verilog reads:
            #   FPGAp/mem/left_eye.hex
            #   FPGAp/mem/right_eye.hex
            #   FPGAp/mem/mouth.hex

            saved_paths = save_fpga_rois(
                rois,
                output_dir=FPGA_MEM_DIR,
                save_txt=True,
                save_hex=True
            )

            # Optional backup copy for manual checking.
            save_fpga_rois_backup(
                rois,
                output_dir=FPGA_ROI_DIR
            )

            # ====================================================
            # Python golden feature generation
            # ====================================================
            # This matches FPGA:
            #   6 fixed kernels
            #   ReLU + clip
            #   MaxPool
            #   sum / max / energy
            #
            # Output:
            #   python_features_left_eye.txt
            #   python_features_right_eye.txt
            #   python_features_mouth.txt
            # ====================================================

            python_features = generate_python_golden_features(
                rois,
                output_dir=FPGA_MEM_DIR
            )

            # ====================================================
            # Fatigue state update
            # ====================================================

            info = state.update(
                ear,
                mar,
                fpga_features=python_features
            )

            now = time.time()

            # ====================================================
            # Periodic logging
            # ====================================================

            if now - last_log_time >= LOG_INTERVAL_SEC:
                logger.add(
                    ear,
                    mar,
                    info,
                    fpga_features=python_features
                )

                logger.save_all()

                print("Saved FPGA ROI inputs to:", FPGA_MEM_DIR)
                print("Saved ROI names:", list(rois.keys()))
                print("Saved paths:", saved_paths)
                print("Saved Python golden features:", list(python_features.keys()))
                print("EAR:", round(ear, 4), "MAR:", round(mar, 4))
                print("State:", info["fatigue_state"])
                print("Attention:", info["attention_score"])

                last_log_time = now

            # ====================================================
            # Draw UI
            # ====================================================

            if DRAW_UI:
                draw_text(frame, f"EAR: {ear:.3f}", 20, 40, color=(0, 255, 255))
                draw_text(frame, f"L_EAR: {left_ear:.3f}", 20, 70, color=(0, 255, 255))
                draw_text(frame, f"R_EAR: {right_ear:.3f}", 20, 100, color=(0, 255, 255))
                draw_text(frame, f"MAR: {mar:.3f}", 20, 130, color=(0, 255, 255))

                draw_text(frame, f"PERCLOS: {info['perclos']:.2f}", 20, 170)
                draw_text(frame, f"Closed: {info['closed']}", 20, 200)
                draw_text(frame, f"CloseDur: {info['current_closure_duration']:.2f}s", 20, 230)

                draw_text(frame, f"Blink: {info['blink_count']}", 20, 260)
                draw_text(frame, f"LongClose: {info['long_closure_count']}", 20, 290)
                draw_text(frame, f"Yawn: {info['yawn_count']}", 20, 320)
                draw_text(frame, f"Microsleep: {info['microsleep_count']}", 20, 350)

                draw_text(
                    frame,
                    f"State: {info['fatigue_state']}",
                    20,
                    390,
                    color=(0, 0, 255),
                    scale=0.9
                )

                draw_text(
                    frame,
                    f"Attention: {info['attention_score']}",
                    20,
                    430,
                    color=(0, 0, 255),
                    scale=0.9
                )

                roi_names_text = ",".join(rois.keys()) if rois else "none"

                draw_text(
                    frame,
                    f"ROIs: {roi_names_text}",
                    20,
                    465,
                    color=(0, 255, 0),
                    scale=0.65
                )

                if python_features:
                    left_summary = summarize_feature_vector(
                        python_features.get("left_eye")
                    )
                    right_summary = summarize_feature_vector(
                        python_features.get("right_eye")
                    )
                    mouth_summary = summarize_feature_vector(
                        python_features.get("mouth")
                    )

                    draw_text(
                        frame,
                        f"L FPGA: {left_summary}",
                        20,
                        495,
                        color=(0, 255, 0),
                        scale=0.45,
                        thickness=1
                    )

                    draw_text(
                        frame,
                        f"R FPGA: {right_summary}",
                        20,
                        520,
                        color=(0, 255, 0),
                        scale=0.45,
                        thickness=1
                    )

                    draw_text(
                        frame,
                        f"M FPGA: {mouth_summary}",
                        20,
                        545,
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