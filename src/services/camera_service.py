from src.utils.video_utils import initialize_video_capture, release_video_capture
from src.utils.hand_utils import initialize_hand_detector, extract_landmarks

class CameraService:
    def __init__(self):
        self.cap = initialize_video_capture(0)
        self.hands = initialize_hand_detector()

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_landmarks(self, frame):
        return extract_landmarks(frame, self.hands)

    def release(self):
        release_video_capture(self.cap)
        self.hands.close()
