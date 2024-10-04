import torch
from src.models.hand_gesture_model import HandGestureModel
from src.utils.data_utils import load_labels

class HandGestureService:
    def __init__(self, model_path, label_file):
        self.labels = load_labels(label_file)
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = HandGestureModel(len(self.labels))
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    def predict(self, landmarks):
        points_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(points_tensor)
            gesture_index = torch.argmax(output, dim=1).item()
        return gesture_index