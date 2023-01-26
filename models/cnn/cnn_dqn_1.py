import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_DQN_1(nn.Module):
    def __init__(self, outputs):
        super(CNN_DQN_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
