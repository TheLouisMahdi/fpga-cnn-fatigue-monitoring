# camera.py

import cv2
from config import VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    def __init__(self, source=VIDEO_SOURCE):
        print("Connecting to:", source)

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera could not be opened: {source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        print("Connected!")
        print(f"Requested frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    def read(self):
        ret, frame = self.cap.read()

        if not ret or frame is None:
            return None

        return frame

    def release(self):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    cam = Camera()

    while True:
        frame = cam.read()

        if frame is None:
            continue

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
