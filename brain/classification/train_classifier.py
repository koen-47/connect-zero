from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .train_test_data import split_train_val_set


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


def train(num_epochs):
    train_data, val_data = split_train_val_set(1)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=True)
    PATH = './connect_4.pth'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    criterion1 = nn.MSELoss()

    model = Classifier1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    loss_interval = 500
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            board_state, result = data
            board_state = board_state.to(device)
            result = result.to(device)

            optimizer.zero_grad()

            output_result = model(board_state).to(device)
            loss = criterion1(output_result, result)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % loss_interval == loss_interval - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_interval:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), PATH)

    correct = 0
    total = 0
    model = Classifier1()
    model.load_state_dict(torch.load(PATH))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct // total}%')

# train(num_epochs=10)
# test()
