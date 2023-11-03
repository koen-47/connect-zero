import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DualConvolutionalNetwork(nn.Module):
    def __init__(self, num_channels):
        super(DualConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=(4, 4), padding=0)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=(2, 2), padding=0)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=(2, 2), padding=0)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(128, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.policy = nn.Linear(512, 7)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=0.3)
        x = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=0.3)

        policy = self.policy(x)
        value = self.value(x)
        return F.softmax(policy, dim=-1), torch.tanh(value)

    def train_on_examples(self, examples, num_epochs=10, lr=0.001, weight_decay=0.0001, logger=None):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()

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
            batch_count = max(1, int(len(examples) / batch_size))
            for i in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=batch_size)
                boards, move, value = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).unsqueeze(dim=1).to(device)
                target_move = torch.FloatTensor(np.array(move)).to(device)
                target_value = torch.FloatTensor(np.array(value).astype(np.float64)).unsqueeze(dim=1).to(device)

                out_move, out_value = self(boards)
                loss_move = criterion1(target_move, out_move)
                loss_value = criterion2(target_value, out_value)
                total_loss = loss_move + loss_value

                # print(target_value, out_value, loss_value)

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
