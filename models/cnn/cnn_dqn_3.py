import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN_CNN_3(nn.Module):
    def __init__(self, num_channels, num_res_blocks, kernel_size=(3, 3), padding=0):
        super(DQN_CNN_3, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(num_channels)
        self.layers = nn.ModuleList()
        for i in range(num_res_blocks):
            self.layers.append(ResidualBlock(num_channels, num_channels, num_channels, kernel_size=(3, 3), padding=1))
        self.policy_head = PolicyHead(num_channels, kernel_size=(3, 3), padding=1)
        self.value_head = ValueHead(num_channels, fc_size=256, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x = F.relu(self.bn2d_1(self.conv1(x)))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.policy_head(x), self.value_head(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, kernel_size=(3, 3), padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(hidden_size)
        self.bn2d_2 = nn.BatchNorm2d(out_size)

    def conv_block(self, x):
        x = F.relu(self.bn2d_1(self.conv1(x)))
        x = F.relu(self.bn2d_2(self.conv2(x)))
        return x

    def forward(self, x):
        return x + self.conv_block(x)


class ValueHead(nn.Module):
    def __init__(self, in_size, fc_size, kernel_size=(3, 3), padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, 1, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(42, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.value = nn.Linear(fc_size, 1)

    def forward(self, x):
        x = F.relu(self.bn2d_1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.value(x))


class PolicyHead(nn.Module):
    def __init__(self, in_size, kernel_size=(2, 2), padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, 1, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(1)
        self.policy = nn.Linear(42, 7)

    def forward(self, x):
        x = F.relu(self.bn2d_1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return F.softmax(self.policy(x), dim=-1)
