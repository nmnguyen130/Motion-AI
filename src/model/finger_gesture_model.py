import torch
import torch.nn as nn
import torch.nn.functional as F

class FingerGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(FingerGestureModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Lớp fully connected
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Kích thước sau khi pool là 32x32
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def conv_block(self, x, conv_layer1, conv_layer2=None):
        x = F.relu(conv_layer1(x))
        if conv_layer2:
            x = F.relu(conv_layer2(x))
        return self.pool(x)

    def forward(self, x):
        x = self.conv_block(x, self.conv1, self.conv2)
        x = self.conv_block(x, self.conv3, self.conv4)
        x = self.conv_block(x, self.conv5, self.conv6)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))  # Thêm dropout
        x = self.fc2(x)
        
        return x