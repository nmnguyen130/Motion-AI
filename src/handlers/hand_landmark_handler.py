import cv2
import mediapipe as mp
import numpy as np

class HandLandmarkHandler:
    def __init__(self, num_hands=1, min_detection_confidence=0.7):
        """
        Quản lý và xử lý các điểm đặc trưng (landmarks) của bàn tay.
        Arguments:
        - num_hands: Số lượng tay cần phát hiện.
        - min_detection_confidence: Độ tự tin tối thiểu để phát hiện bàn tay.
        """
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.num_hands,
            min_detection_confidence=self.min_detection_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks(self, frame, isDraw=True):
        """
        Trích xuất đặc trưng từ frame của ảnh.
        Arguments:
        - frame: Mảng numpy chứa dữ liệu hình ảnh (BGR) từ camera hoặc file ảnh.
        - isDraw: Có vẽ các điểm đặc trưng lên frame hay không.

        Returns:
        - List các mảng numpy chứa tọa độ các điểm đặc trưng của mỗi bàn tay hoặc None nếu không tìm thấy bàn tay.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)

        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if isDraw:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Lưu tọa độ đặc trưng của từng bàn tay
                points = []
                for landmark in hand_landmarks.landmark:
                    points.extend([landmark.x, landmark.y, landmark.z])
                landmarks_list.append(np.array(points))
        return landmarks_list if landmarks_list else None

    