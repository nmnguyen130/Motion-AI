import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

from .model.finger_gesture_model import FingerGestureModel  # Thay đổi đường dẫn nếu cần

class FingerGestureRecognizer:
    def __init__(self, model_path, num_classes):
        self.model = FingerGestureModel(num_classes)
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))  # Đảm bảo nạp mô hình đúng cách
        self.model.eval()  # Đặt mô hình ở chế độ đánh giá
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')  # Đảm bảo hình ảnh là RGB
        img = self.transform(img).unsqueeze(0)  # Thêm kích thước batch
        with torch.no_grad():
            outputs = self.model(img)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()  # Trả về chỉ số lớp

def main():
    model_path = 'src/model/finger_gesture_model.pth'  # Đường dẫn đến mô hình
    test_image_path = 'src/data/test/index_up.webp'  # Đường dẫn đến hình ảnh cần kiểm tra
    recognizer = FingerGestureRecognizer(model_path, num_classes=3)

    gesture_index = recognizer.predict(test_image_path)
    print(f'Predicted gesture index: {gesture_index}')

if __name__ == "__main__":
    main()