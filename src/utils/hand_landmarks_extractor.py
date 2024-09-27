import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(frame):
    """
    Trích xuất 21 điểm đặc trưng từ frame của ảnh.
    Arguments:
    - frame: Mảng numpy chứa dữ liệu hình ảnh (BGR) từ camera hoặc file ảnh.

    Returns:
    - points: Mảng numpy chứa tọa độ các điểm đặc trưng hoặc None nếu không tìm thấy bàn tay.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            for landmark in hand_landmarks.landmark:
                points.extend([landmark.x, landmark.y, landmark.z])
            return np.array(points)
    return None
