from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from train_test_data import split_train_val_set


class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv7 = nn.Conv2d(64, 3, kernel_size=(1, 1))
        self.fc1 = nn.Linear(2688, 64)
        self.value = nn.Linear(64, 1)
        self.policy = nn.Linear(64, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # return torch.tanh(self.value(x))
        # return F.tanh(self.value(x)), F.softmax(self.policy(x), dim=-1)
        return F.log_softmax(self.policy(x), dim=-1)


def train_model(player_id, num_epochs):
    train_data, val_data = split_train_val_set(player_id)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=True)
    PATH = f'./classification_model_p{player_id}.pth'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    criterion1 = nn.MSELoss()
    criterion2 = nn.NLLLoss()

    model = Classifier1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_running_acc = 0.0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            model.train()
            board_state, optimal_move = data
            board_state = board_state.to(device)
            optimal_move = optimal_move.to(device)

            optimizer.zero_grad()

            output_optimal_move = model(board_state).to(device)
            train_loss = criterion2(output_optimal_move, optimal_move)

            train_loss.backward()
            optimizer.step()

            train_running_loss += train_loss.item()
            _, train_predicted = torch.max(output_optimal_move.data, 1)
            train_total += optimal_move.size(0)
            train_running_acc += (train_predicted == optimal_move).sum().item()

        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        val_total = 0
        for j, val_data in enumerate(val_loader, 0):
            val_board_state, val_optimal_move = val_data
            val_board_state = val_board_state.to(device)
            val_optimal_move = val_optimal_move.to(device)
            val_output = model(val_board_state).to(device)
            val_loss = criterion2(val_output, val_optimal_move)

            val_running_loss += val_loss.item()
            _, val_predicted = torch.max(val_output.data, 1)
            val_total += val_optimal_move.size(0)
            val_running_acc += (val_predicted == val_optimal_move).sum().item()

        print(f'[{epoch + 1}]   '
              f'train_loss: {train_running_loss / len(train_loader):.3f}, '
              f'train_acc: {train_running_acc / train_total:.3f}, '
              f'val_loss: {val_running_loss / len(val_loader):.3f}, '
              f'val_acc: {val_running_acc / val_total:.3f}')

    torch.save(model.state_dict(), PATH)

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in val_loader:
    #         images, labels = data
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print(f'accuracy: {100 * correct // total}%')


train_model(player_id=2, num_epochs=10)

