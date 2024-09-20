import cv2

def initialize_video_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    return cap

def release_video_capture(cap):
    cap.release()
    cv2.destroyAllWindows()