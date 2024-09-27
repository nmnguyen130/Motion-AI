import cv2
import torch
import numpy as np
import mediapipe as mp
from src.models.hand_gesture_model import HandGestureModel
from src.utils.video_utils import initialize_video_capture, release_video_capture

def load_model(model_path, num_classes=3):
    model = HandGestureModel(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def initialize_hand_detector(min_detection_confidence=0.7):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence
    )
    return hands, mp.solutions.drawing_utils

def predict_gesture(model, landmarks):
    points_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(points_tensor)
        gesture = torch.argmax(output, dim=1).item()
    return gesture

def draw_landmarks_and_label(frame, hand_landmarks, gesture, mp_drawing, mp_hands):
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def main():
    model_path = 'src/models/hand_gesture_model_camera.pth'
    model = load_model(model_path)
    hands, mp_drawing = initialize_hand_detector()

    cap = initialize_video_capture(0)

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
                landmarks = [landmark for point in hand_landmarks.landmark for landmark in (point.x, point.y, point.z)]
                
                # Dự đoán cử chỉ và hiển thị kết quả
                gesture = predict_gesture(model, landmarks)
                draw_landmarks_and_label(frame, hand_landmarks, gesture, mp_drawing, mp.solutions.hands)

        cv2.imshow('Hand Gesture Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    release_video_capture(cap)
    hands.close()

if __name__ == "__main__":
    main()
