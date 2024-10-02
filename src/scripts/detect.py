import cv2
import torch
import numpy as np
import mediapipe as mp
from src.models.hand_gesture_model import HandGestureModel
from src.utils.data_utils import load_labels
from src.utils.video_utils import initialize_video_capture, release_video_capture
from src.utils.hand_utils import initialize_hand_detector, draw_label, extract_landmarks

def load_model(model_path, num_classes=3):
    model = HandGestureModel(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict_gesture(model, landmarks):
    points_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(points_tensor)
        gesture_index = torch.argmax(output, dim=1).item()
    return gesture_index

def main():
    labels = load_labels('src/data/processed/hand_gesture_labels.csv')
    reversed_labels = {value: key for key, value in labels.items()}

    model_path = 'src/models/hand_gesture_model_camera.pth'
    model = load_model(model_path, len(labels))

    hands = initialize_hand_detector()

    cap = initialize_video_capture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame, hands)
        
        if landmarks is not None:
            gesture_index = predict_gesture(model, landmarks)
            gesture_name = reversed_labels.get(gesture_index, "Unknown")
            draw_label(frame, gesture_name)

        cv2.imshow('Hand Gesture Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    release_video_capture(cap)
    hands.close()

if __name__ == "__main__":
    main()
