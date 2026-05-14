# fatigue_state.py

import time
from collections import deque

from config import (
    EAR_CLOSED_THRESHOLD,
    EAR_OPEN_THRESHOLD,
    MAR_YAWN_THRESHOLD,
    YAWN_MIN_DURATION,
    MICROSLEEP_MIN_DURATION,
    PERCLOS_WINDOW_SEC,
    PERCLOS_FATIGUE_THRESHOLD,
)


class FatigueState:
    """
    Tracks fatigue-related temporal behavior.

    Inputs:
        ear:
            Eye Aspect Ratio.
            Smaller value usually means more closed eyes.

        mar:
            Mouth Aspect Ratio.
            Larger value usually means more open mouth.

        fpga_features:
            Optional dictionary or feature vector from FPGA.
            Not used directly in final decision yet.
            Kept for future integration.

    Output:
        Dictionary with fatigue state, counters, PERCLOS, and attention score.
    """

    def __init__(self):
        # ------------------------------
        # Eye state
        # ------------------------------
        self.eye_closed = False
        self.eye_close_start = None

        # ------------------------------
        # Event counters
        # ------------------------------
        self.blink_count = 0
        self.long_closure_count = 0
        self.microsleep_count = 0

        self.yawn_active = False
        self.yawn_start = None
        self.yawn_count = 0

        # ------------------------------
        # Time windows
        # ------------------------------
        self.perclos_window = deque()
        self.yawn_window = deque()
        self.long_closure_window = deque()

        # ------------------------------
        # Last values for logging/debug
        # ------------------------------
        self.last_ear = 0.0
        self.last_mar = 0.0
        self.last_fpga_features = None

    def update(self, ear, mar, fpga_features=None):
        """
        Update fatigue state using current EAR and MAR.

        This function is backward-compatible with the old main.py:

            info = state.update(ear, mar)

        It also supports future FPGA feature input:

            info = state.update(ear, mar, fpga_features=features)
        """

        now = time.time()

        self.last_ear = float(ear)
        self.last_mar = float(mar)
        self.last_fpga_features = fpga_features

        # ============================================================
        # Eye closed detection with hysteresis
        # ============================================================
        # If eye is already closed, use EAR_OPEN_THRESHOLD to decide
        # when it becomes open again.
        #
        # If eye is open, use EAR_CLOSED_THRESHOLD to decide when it
        # becomes closed.
        #
        # This reduces rapid switching around one threshold.
        # ============================================================

        if self.eye_closed:
            closed = ear < EAR_OPEN_THRESHOLD
        else:
            closed = ear < EAR_CLOSED_THRESHOLD

        # ============================================================
        # PERCLOS
        # ============================================================
        # PERCLOS means percentage of time/frames that eyes are closed
        # inside a recent time window.
        # ============================================================

        self.perclos_window.append((now, 1 if closed else 0))

        while self.perclos_window and now - self.perclos_window[0][0] > PERCLOS_WINDOW_SEC:
            self.perclos_window.popleft()

        perclos = self.get_perclos()

        # ============================================================
        # Blink / long closure / microsleep
        # ============================================================

        current_closure_duration = 0.0

        # Eye just became closed
        if closed and not self.eye_closed:
            self.eye_close_start = now

        # Eye is still closed
        if closed and self.eye_close_start is not None:
            current_closure_duration = now - self.eye_close_start

        # Eye just opened
        if not closed and self.eye_closed:
            duration = now - self.eye_close_start if self.eye_close_start else 0.0

            if 0.08 <= duration < 0.45:
                self.blink_count += 1

            elif 0.45 <= duration < 1.5:
                self.long_closure_count += 1
                self.long_closure_window.append(now)

            elif duration >= 1.5:
                self.microsleep_count += 1
                self.long_closure_window.append(now)

            self.eye_close_start = None

        self.eye_closed = closed

        # Remove old long-closure events
        while self.long_closure_window and now - self.long_closure_window[0] > 60.0:
            self.long_closure_window.popleft()

        # ============================================================
        # Yawn detection
        # ============================================================
        # If MAR stays above threshold for enough time, count as yawn.
        # ============================================================

        if mar >= MAR_YAWN_THRESHOLD and not self.yawn_active:
            self.yawn_active = True
            self.yawn_start = now

        if mar < MAR_YAWN_THRESHOLD and self.yawn_active:
            duration = now - self.yawn_start if self.yawn_start else 0.0

            if duration >= YAWN_MIN_DURATION:
                self.yawn_count += 1
                self.yawn_window.append(now)

            self.yawn_active = False
            self.yawn_start = None

        # Remove old yawn events
        while self.yawn_window and now - self.yawn_window[0] > 120.0:
            self.yawn_window.popleft()

        # ============================================================
        # Optional FPGA status
        # ============================================================
        # For now, FPGA features are only passed through and marked.
        # Later we can use them inside classify_state().
        # ============================================================

        fpga_available = fpga_features is not None

        # ============================================================
        # Final state decision
        # ============================================================

        fatigue_state = self.classify_state(
            perclos=perclos,
            yawn_recent=len(self.yawn_window),
            long_closure_recent=len(self.long_closure_window),
            current_closure_duration=current_closure_duration,
            fpga_features=fpga_features,
        )

        attention_score = self.calculate_attention_score(
            perclos=perclos,
            yawn_recent=len(self.yawn_window),
            long_closure_recent=len(self.long_closure_window),
            current_closure_duration=current_closure_duration,
            state=fatigue_state,
            fpga_features=fpga_features,
        )

        return {
            "closed": closed,
            "current_closure_duration": round(current_closure_duration, 2),
            "perclos": round(perclos, 4),

            "blink_count": self.blink_count,
            "long_closure_count": self.long_closure_count,
            "yawn_count": self.yawn_count,
            "microsleep_count": self.microsleep_count,

            "yawn_recent": len(self.yawn_window),
            "long_closure_recent": len(self.long_closure_window),

            "fatigue_state": fatigue_state,
            "attention_score": attention_score,

            "fpga_available": fpga_available,
        }

    def get_perclos(self):
        """
        Calculate PERCLOS from the current time window.
        """

        if not self.perclos_window:
            return 0.0

        closed_sum = sum(v for _, v in self.perclos_window)
        return closed_sum / len(self.perclos_window)

    def classify_state(
        self,
        perclos,
        yawn_recent,
        long_closure_recent,
        current_closure_duration,
        fpga_features=None,
    ):
        """
        Classify fatigue state.

        Current decision is mainly based on:
            PERCLOS
            current closure duration
            recent yawns
            recent long closures

        FPGA features are accepted but not used strongly yet.
        They will be integrated after Python golden verification.
        """

        # Strong immediate warning
        if current_closure_duration >= 2.0:
            return "MICROSLEEP_WARNING"

        # Strong fatigue by PERCLOS
        if perclos >= 0.35:
            return "FATIGUE"

        # Config threshold support
        if perclos >= max(PERCLOS_FATIGUE_THRESHOLD, 0.25) and (
            long_closure_recent >= 2 or yawn_recent >= 2
        ):
            return "FATIGUE"

        # Drowsy state
        if perclos >= 0.18 or long_closure_recent >= 2 or yawn_recent >= 2:
            return "DROWSY"

        return "ALERT"

    def calculate_attention_score(
        self,
        perclos,
        yawn_recent,
        long_closure_recent,
        current_closure_duration,
        state,
        fpga_features=None,
    ):
        """
        Calculate attention score from 0 to 100.

        Higher score means better attention.
        Lower score means more fatigue or distraction risk.
        """

        score = 100.0

        # PERCLOS penalty
        if perclos > 0.10:
            score -= (perclos - 0.10) * 80.0

        # Recent yawn penalty
        score -= min(yawn_recent, 4) * 4.0

        # Recent long eye closure penalty
        score -= min(long_closure_recent, 4) * 6.0

        # Current eye closure duration penalty
        if current_closure_duration > 0.5:
            score -= min((current_closure_duration - 0.5) * 10.0, 20.0)

        # State-level correction
        if state == "DROWSY":
            score -= 8.0
        elif state == "FATIGUE":
            score -= 18.0
        elif state == "MICROSLEEP_WARNING":
            score -= 25.0

        # Reserved place for future FPGA penalty/bonus.
        # For now, no direct penalty is applied because FPGA features
        # still need golden-model validation.
        if fpga_features is not None:
            score -= 0.0

        return round(max(0.0, min(100.0, score)), 1)

    def reset(self):
        """
        Reset all counters and states.
        Useful when starting a new video or new subject.
        """

        self.__init__()