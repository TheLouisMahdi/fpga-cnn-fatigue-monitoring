# camera.py

import cv2
from config import VIDEO_SOURCE


class Camera:
    def __init__(self, source=VIDEO_SOURCE):
        print("Connecting to:", source)

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera could not be opened: {source}")

        print("Connected!")

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

        cv2.imshow("IP Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()