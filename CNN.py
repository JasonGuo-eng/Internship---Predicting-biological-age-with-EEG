#1D CNN model
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG1DCNN(nn.Module):
    def __init__(self):
        super(EEG1DCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=7, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveMaxPool1d(10)  # Output shape: [B, 128, 10]

        self.fc1 = nn.Linear(64 * 10, 32)
        self.dropout = nn.Dropout(0.3) #0.3 is better than 0.2 and 0.5
        self.fc2 = nn.Linear(32, 1)  # Regression output

    def forward(self, x):
        # Input shape: [B, 1, 10, 179]
        x = x.squeeze(1)  # Shape: [B, 10, 179]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
