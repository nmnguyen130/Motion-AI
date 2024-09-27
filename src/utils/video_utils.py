import cv2

def initialize_video_capture(device_index=0):
    """
    Khởi tạo và kiểm tra camera từ device index.
    
    Parameters:
    - device_index: Chỉ số của thiết bị video (mặc định là 0 cho webcam chính).
    
    Returns:
    - cap: Đối tượng VideoCapture đã được mở.
    """
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"Error: Could not open video device with index {device_index}.")
        exit()
    return cap

def release_video_capture(cap):
    """
    Giải phóng tài nguyên video và đóng tất cả các cửa sổ OpenCV.
    
    Parameters:
    - cap: Đối tượng VideoCapture cần giải phóng.
    """
    cap.release()
    cv2.destroyAllWindows()