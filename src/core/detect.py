import cv2
from src.services.hand_gesture_service import HandGestureService
from src.services.camera_service import CameraService
from src.utils.hand_utils import draw_label

def main():
    model_path = 'src/models/hand_gesture_model_camera.pth'
    label_file = 'src/configs/labels.csv'
    recognition_service = HandGestureService(model_path, label_file)

    camera_service = CameraService()

    while True:
        frame = camera_service.capture_frame()
        if frame is None:
            break

        landmarks = camera_service.get_landmarks(frame)
        if landmarks is not None:
            gesture_index = recognition_service.predict(landmarks)
            draw_label(frame, recognition_service.labels[gesture_index])

        cv2.imshow('Hand Gesture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_service.release()

if __name__ == "__main__":
    main()
