import torch
import torch.nn as nn
import torch.nn.functional as F

class HandGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(HandGestureModel, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x