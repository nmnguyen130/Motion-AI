import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from .model.finger_gesture_model import FingerGestureModel

# Cấu hình
train_transforms = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=20),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Chỉ số cho ảnh grayscale
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Chỉ số cho ảnh grayscale
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset = datasets.ImageFolder(root='src/data/train', transform=train_transforms)
val_dataset = datasets.ImageFolder(root='src/data/validation', transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

# Model, criterion, optimizer
model = FingerGestureModel(num_classes=len(train_dataset.classes)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training functions
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Tính độ chính xác
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
    
    train_accuracy = 100 * correct / total
    return running_loss / len(train_loader), train_accuracy

# Validation functions
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss += criterion(val_outputs, val_labels).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    return val_loss / len(val_loader), val_accuracy

# Training loop
num_epochs = 50
history = []

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)
    
    history.append((train_loss, train_accuracy, val_loss, val_accuracy))

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%')

# Lưu mô hình
torch.save(model.state_dict(), 'src/model/finger_gesture_model.pth')