import cv2
import torch
import mediapipe as mp
from models.hand_gesture_model import HandGestureModel
import numpy as np

# Tải mô hình
model = HandGestureModel()
model.load_state_dict(torch.load('../models/hand_gesture_model.pth'))
model.eval()

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Bật camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi ảnh sang RGB để xử lý
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Nếu phát hiện bàn tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            for landmark in hand_landmarks.landmark:
                points.extend([landmark.x, landmark.y, landmark.z])
            
            # Chuyển điểm đặc trưng thành tensor
            points = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
            
            # Dự đoán cử chỉ
            with torch.no_grad():
                output = model(points)
                gesture = torch.argmax(output, dim=1).item()
                cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Vẽ các điểm trên tay
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture Detection', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
