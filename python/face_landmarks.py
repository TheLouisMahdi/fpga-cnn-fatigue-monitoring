# face_landmarks.py

import cv2
import mediapipe as mp


class FaceLandmarks:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        return result

    def draw(self, frame, result):
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1,
                        circle_radius=1
                    )
                )
        return frame

    def close(self):
        self.face_mesh.close()