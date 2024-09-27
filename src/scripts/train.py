import torch
import torch.nn as nn
import torch.optim as optim
from models.hand_gesture_model import HandGestureModel
from utils.data_loader import load_data

# Khởi tạo nhãn cử chỉ
labels = {'open': 0, 'closed': 1, 'fist': 2, 'thumbs_up': 3, 'peace': 4}
processed_dir = '../data/processed'

# Load dữ liệu từ file CSV
data, target = load_data(processed_dir, labels)

# Khởi tạo model, loss function và optimizer
model = HandGestureModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Lưu mô hình
torch.save(model.state_dict(), '../models/hand_gesture_model.pth')