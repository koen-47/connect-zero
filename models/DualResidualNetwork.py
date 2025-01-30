import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DualResidualNetwork(nn.Module):
    def __init__(self, num_channels, num_res_blocks, kernel_size=(3, 3), padding=1):
        super(DualResidualNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(num_channels)
        self.layers = nn.ModuleList()
        for i in range(num_res_blocks):
            self.layers.append(ResidualBlock(num_channels, num_channels, num_channels, kernel_size=(3, 3), padding=1))
        self.policy_head = PolicyHead(num_channels, kernel_size=(1, 1), padding=1)
        self.value_head = ValueHead(num_channels, fc_size=128, kernel_size=(1, 1), padding=1)

    def forward(self, x):
        x = F.relu(self.bn2d_1(self.conv1(x)))
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=0.3, training=self.training)
        return self.policy_head(x), self.value_head(x)

    def train_on_examples(self, examples, num_epochs=10, lr=0.001, weight_decay=0.0001, logger=None):
        shuffled_examples = copy.deepcopy(examples)
        np.random.shuffle(shuffled_examples)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.to(device)

        for epoch in range(num_epochs):
            model.train()

            sum_total_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0
            sum_policy_acc = 0.0
            total_policy_acc = 0

            batch_size = 64
            batch_count = max(1, int(len(shuffled_examples) / batch_size))
            for i in range(batch_count):
                sample_ids = np.random.randint(len(shuffled_examples), size=batch_size)
                boards, move, value = list(zip(*[shuffled_examples[j] for j in sample_ids]))
                boards = torch.tensor(np.array(boards)).float().to(device)

                target_move = torch.tensor(np.array(move)).float().to(device)
                target_value = torch.tensor(np.array(value)).float().to(device)

                out_move, out_value = self(boards)
                loss_move = self.loss_pi(target_move, out_move)
                loss_value = self.loss_v(target_value, out_value)
                total_loss = loss_move + loss_value

                sum_total_loss += total_loss.item()
                sum_policy_loss += loss_move.item()
                sum_value_loss += loss_value.item()
                total_policy_acc += out_move.size(0)
                sum_policy_acc += (torch.sum(torch.argmax(target_move, dim=1) == torch.argmax(out_move, dim=1))).item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            total_loss = sum_total_loss / batch_count
            value_loss = sum_value_loss / batch_count
            policy_acc = sum_policy_acc / total_policy_acc
            print(f"Epoch {epoch + 1}. "
                  f"Total loss: {total_loss:.3f}. "
                  f"Value loss: {value_loss:.3f}. "
                  f"Policy acc.: {policy_acc:.3f}")
            logger.log(f"(Training) Epoch: {epoch + 1}. Total loss: {total_loss:.3f}. Value loss: {value_loss:.3f}. "
                       f"Policy accuracy: {policy_acc:.3f}", to_summary=True, to_iteration=True)
        return copy.deepcopy(self)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


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
    def __init__(self, in_size, fc_size, kernel_size=(1, 1), padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, 32, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(2304, fc_size)
        self.value = nn.Linear(fc_size, 1)

    def forward(self, x):
        x = F.relu(self.bn2d_1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.value(x))


class PolicyHead(nn.Module):
    def __init__(self, in_size, kernel_size=(1, 1), padding=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, 32, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.policy = nn.Linear(2304, 7)

    def forward(self, x):
        x = F.relu(self.bn2d_1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.policy(x)
        return F.log_softmax(x, dim=1)
