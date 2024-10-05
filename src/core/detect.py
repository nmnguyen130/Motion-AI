import cv2
from src.services.hand_gesture_service import HandGestureService
from src.services.camera_service import CameraService
from src.utils.visualization_utils import draw_label

def main():
    model_path = 'experiments/hand_gesture_exp/hand_gesture_model_camera.pth'
    label_file = 'data/processed/hand_gesture_labels.csv'

    recognition_service = HandGestureService(model_path, label_file)
    camera_service = CameraService(num_hands=2)

    while True:
        frame = camera_service.capture_frame()
        if frame is None:
            break

        landmarks_list = camera_service.get_landmarks(frame)

        if landmarks_list:
            for i, landmarks in enumerate(landmarks_list):
                hand_position = 'left' if i == 0 else 'right'

                gesture_index = recognition_service.predict(landmarks)
                gesture_name = recognition_service.get_gesture_name(gesture_index)

                draw_label(frame, gesture_name, hand_position)

        cv2.imshow('Hand Gesture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_service.release()

if __name__ == "__main__":
    main()
