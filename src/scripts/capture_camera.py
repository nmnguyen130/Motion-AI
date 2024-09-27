import os
import cv2
import pandas as pd
from src.utils.file_utils import create_directory_if_not_exists
from src.utils.video_utils import initialize_video_capture, release_video_capture
from src.utils.data_utils import load_labels, initialize_dataframe, save_dataframe_to_csv
from src.utils.hand_landmarks_extractor import extract_landmarks

def capture_from_camera(csv_output_path):
    """
    Capture data from camera, extract landmarks, and save to a CSV file.
    - Press 's' to capture data.
    - Press 'q' to quit.
    """

    create_directory_if_not_exists(os.path.dirname(csv_output_path))

    # Khởi tạo camera và các biến cần thiết
    cap = initialize_video_capture(0)
    
    # Tạo dataframe để lưu điểm đặc trưng
    columns = [f'point_{i}_{axis}' for i in range(1, 22) for axis in ['x', 'y', 'z']] + ['label']  # Tổng cộng 64 cột
    df = initialize_dataframe(columns, csv_output_path)
    current_label = None

    label_map = {
        '1': 'fist',
        '2': 'five',
        '3': 'thumb_up'
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_label:
            cv2.putText(frame, f"Current Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị khung hình lên cửa sổ camera
        cv2.imshow('Capture Hand Gesture', frame)

        # Nhấn phím để điều khiển
        key = cv2.waitKey(1) & 0xFF

        # Nhấn phím số (1-3) để chọn nhãn cho dữ liệu
        if chr(key) in label_map:
            current_label = label_map[chr(key)]
            print(f"Nhãn hiện tại: {current_label}")

        # Nhấn 's' để chụp ảnh và ghi dữ liệu
        elif key == ord('s'):
            if current_label:
                landmarks = extract_landmarks(frame)  # Trích xuất 21 điểm đặc trưng từ frame
                if landmarks is not None and len(landmarks) == 63:
                    landmarks = landmarks.tolist()
                    landmarks.append(current_label)
                    # Lưu vào DataFrame
                    df = pd.concat([df, pd.DataFrame([landmarks], columns=df.columns)], ignore_index=True)
                    print(f"Đã ghi dữ liệu cho nhãn: {current_label}")

        # Nhấn 'q' để thoát
        elif key == ord('q'):
            break

    save_dataframe_to_csv(df, csv_output_path)
    
    release_video_capture(cap)

if __name__ == "__main__":
    capture_from_camera('src/data/processed/hand_gesture_camera.csv')
