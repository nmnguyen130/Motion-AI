import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.hand_gesture_model import HandGestureModel
from src.utils.data_utils import load_labels, load_data

def train_from_camera(csv_file, model_save_path):
    # Load csv
    labels = load_labels('src/data/processed/hand_gesture_labels.csv')
    data, targets = load_data(csv_file, labels)
    
    # Tạo DataLoader
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Khởi tạo mô hình
    model = HandGestureModel(len(labels))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Huấn luyện mô hình
    for epoch in range(60):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    # Lưu mô hình đã huấn luyện
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    # Huấn luyện từ file CSV chứa dữ liệu từ camera
    train_from_camera('src/data/processed/hand_gesture_camera.csv', 'src/models/hand_gesture_model_camera.pth')
