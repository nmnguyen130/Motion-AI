import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from .finger_detector import FingerDetector
from .model.finger_gesture_model import FingerGestureModel
from .utils.video_utils import initialize_video_capture, release_video_capture

class FingerGestureRecognizer:
    def __init__(self, model_path, num_classes):
        self.model = FingerGestureModel(num_classes)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def predict(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = self.transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()  # Return the class index

def main():
    cap = initialize_video_capture()
    detector = FingerDetector()
    recognizer = FingerGestureRecognizer('src/model/finger_gesture_model.pth', num_classes=3)

    gesture_labels = {0: 'Fist', 1: 'Index Finger Up', 2: 'Thumb Up'}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            hand_frame, hand_detected = detector.detect_fingers(frame)

            if hand_detected:  # Chỉ dự đoán nếu tìm thấy bàn tay
                gesture_index = recognizer.predict(hand_frame)
                gesture = gesture_labels.get(gesture_index, 'Unknown Gesture')
                cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, 'No gesture', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Finger Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        release_video_capture(cap)

if __name__ == "__main__":
    main()