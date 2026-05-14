# logger.py

import os
import pandas as pd
from datetime import datetime

from config import OUTPUT_CSV, OUTPUT_XLSX


# ============================================================
# FPGA feature names
# ============================================================
# Each ROI has 18 features:
# 6 kernels x 3 summary values
#
# Order must match:
#   golden_model.py
#   cnn_feature_extractor_all_top.v
#   fpga_features_*.txt
# ============================================================

FEATURE_NAMES = [
    "vertical_sum",
    "vertical_max",
    "vertical_energy",

    "horizontal_sum",
    "horizontal_max",
    "horizontal_energy",

    "diag45_sum",
    "diag45_max",
    "diag45_energy",

    "diag135_sum",
    "diag135_max",
    "diag135_energy",

    "sharpen_sum",
    "sharpen_max",
    "sharpen_energy",

    "center_sum",
    "center_max",
    "center_energy",
]

ROI_NAMES = [
    "left_eye",
    "right_eye",
    "mouth",
]


class DataLogger:
    """
    Data logger for fatigue monitoring.

    It stores:
        - EAR
        - MAR
        - PERCLOS
        - blink/yawn/microsleep counters
        - fatigue state
        - attention score
        - optional FPGA features

    The logger is backward-compatible.

    Old usage:
        logger.add(ear, mar, info)

    New usage:
        logger.add(ear, mar, info, fpga_features=features)
    """

    def __init__(self):
        self.rows = []

    def add(self, ear, mar, info, fpga_features=None):
        """
        Add one log row.

        Parameters:
            ear:
                Eye Aspect Ratio

            mar:
                Mouth Aspect Ratio

            info:
                Dictionary returned by FatigueState.update()

            fpga_features:
                Optional FPGA feature data.

                Recommended format:
                    {
                        "left_eye":  [18 numbers],
                        "right_eye": [18 numbers],
                        "mouth":     [18 numbers]
                    }

                If fpga_features is None, FPGA columns are filled with None.
        """

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "EAR": round(float(ear), 4),
            "MAR": round(float(mar), 4),

            "PERCLOS": round(float(info.get("perclos", 0.0)), 4),

            "closed": info.get("closed", None),
            "current_closure_duration": info.get("current_closure_duration", None),

            "blink_count": info.get("blink_count", 0),
            "long_closure_count": info.get("long_closure_count", 0),
            "yawn_count": info.get("yawn_count", 0),
            "microsleep_count": info.get("microsleep_count", 0),

            "yawn_recent": info.get("yawn_recent", None),
            "long_closure_recent": info.get("long_closure_recent", None),

            "fatigue_state": info.get("fatigue_state", "UNKNOWN"),
            "attention_score": info.get("attention_score", None),

            "fpga_available": info.get("fpga_available", fpga_features is not None),
        }

        self._add_fpga_features_to_row(row, fpga_features)

        self.rows.append(row)

    def _add_fpga_features_to_row(self, row, fpga_features):
        """
        Add FPGA feature columns to log row.

        Expected fpga_features format:
            {
                "left_eye":  [18 values],
                "right_eye": [18 values],
                "mouth":     [18 values]
            }

        If a ROI or feature is missing, None is written.
        """

        for roi_name in ROI_NAMES:
            values = None

            if isinstance(fpga_features, dict):
                values = fpga_features.get(roi_name)

            for i, feature_name in enumerate(FEATURE_NAMES):
                column_name = f"fpga_{roi_name}_{feature_name}"

                if values is not None and i < len(values):
                    row[column_name] = int(values[i])
                else:
                    row[column_name] = None

    def save_csv(self, path=OUTPUT_CSV):
        """
        Save logs as CSV.
        """
        if not self.rows:
            return

        output_dir = os.path.dirname(path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)

    def save_excel(self, path=OUTPUT_XLSX):
        """
        Save logs as Excel.
        """
        if not self.rows:
            return

        output_dir = os.path.dirname(path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(self.rows)
        df.to_excel(path, index=False)

    def save_all(self):
        """
        Save both CSV and Excel logs.
        """
        if not self.rows:
            return

        self.save_csv()
        self.save_excel()

        print("Logs saved:", OUTPUT_CSV, OUTPUT_XLSX)

    def clear(self):
        """
        Clear all stored rows.
        """
        self.rows = []

    def get_dataframe(self):
        """
        Return current logs as pandas DataFrame.
        Useful for debugging.
        """
        return pd.DataFrame(self.rows)