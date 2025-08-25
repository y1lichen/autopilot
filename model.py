import torch
import torch.nn as nn
import torch.nn.functional as F

class LaneNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),  # -> (32,H/2,W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32,H/4,W/4)

            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # -> (64,H/4,W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64,H/8,W/8)

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # -> (128,H/8,W/8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (128,H/16,W/16)

            nn.Conv2d(128, 256, 3, stride=1, padding=1), # -> (256,H/16,W/16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # 使用 AdaptiveAvgPool2d 前確保 H,W 可整除 4
        self.pool = nn.AdaptiveAvgPool2d((8,8))  # 這樣 256*8*8=16384

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
