import os
import cv2
import pandas as pd
from src.utils.hand_landmarks_extractor import extract_landmarks

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def capture_from_camera(csv_output_path):
    """
    Capture data from camera, extract landmarks, and save to a CSV file.
    - Press 's' to capture data.
    - Press 'q' to quit.
    """

    create_directory_if_not_exists(os.path.dirname(csv_output_path))

    # Khởi tạo camera và các biến cần thiết
    cap = cv2.VideoCapture(0)
    
    # Tạo dataframe để lưu điểm đặc trưng
    columns = [f'point_{i}_{axis}' for i in range(1, 22) for axis in ['x', 'y', 'z']] + ['label']  # Tổng cộng 64 cột
    df = pd.DataFrame(columns=columns)
    current_label = None

    # Kiểm tra xem file CSV đã tồn tại chưa và đọc dữ liệu vào DataFrame
    if os.path.exists(csv_output_path):
        df = pd.read_csv(csv_output_path)
    else:
        df = pd.DataFrame(columns=columns)

    label_map = {
        '1': 'Fist',
        '2': 'Thumb_up',
        '3': 'Thumb_down'
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

    # Lưu dữ liệu vào file CSV
    df.to_csv(csv_output_path, index=False)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_from_camera('src/data/processed/hand_gesture_from_camera.csv')
