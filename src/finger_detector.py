import cv2
import mediapipe as mp

class FingerDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_fingers(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Tạo vùng xung quanh bàn tay
                margin = 20  # Độ rộng vùng xung quanh
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, w)
                y_max = min(y_max + margin, h)

                # Cắt vùng ảnh xung quanh bàn tay
                hand_frame = frame[y_min:y_max, x_min:x_max]

            return hand_frame, True
        return frame, False