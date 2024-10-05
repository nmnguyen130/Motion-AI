import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.hand_gesture_model import HandGestureModel
from src.utils.data_utils import load_labels, load_data

def train_model(csv_file, model_save_path):
    labels = load_labels('data/processed/hand_gesture_labels.csv')
    data, targets = load_data(csv_file, labels)

    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = HandGestureModel(len(labels))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    train_model('data/processed/hand_gesture_camera.csv', 'experiments/hand_gesture_exp/hand_gesture_model_camera.pth')
