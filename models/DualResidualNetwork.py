import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DualResidualNetwork(nn.Module):
    """
    PyTorch model for the dual residual network.
    """
    def __init__(self, num_channels, num_res_blocks, kernel_size=(3, 3), padding=1):
        """
        Constructor for the residual network.
        :param num_channels: number of channels/filters per convolutional layer for the residual blocks, policy head
        and value head.
        :param num_res_blocks: number of residual blocks in the model.
        :param kernel_size: kernel size per convolutional layer for each residual block.
        :param padding: padding per convolutional layer for each residual block.
        """
        super(DualResidualNetwork, self).__init__()

        # First conv. layer + batch normalization.
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(num_channels)

        # Add sequence of residual blocks.
        self.layers = nn.ModuleList()
        for i in range(num_res_blocks):
            self.layers.append(ResidualBlock(num_channels, num_channels, num_channels, kernel_size=(3, 3), padding=1))

        # Add policy head.
        self.policy_head = PolicyHead(num_channels, kernel_size=(1, 1), padding=1)

        # Add value head.
        self.value_head = ValueHead(num_channels, fc_size=128, kernel_size=(1, 1), padding=1)

    def forward(self, x):
        """
        Forward function for dual residual network.
        :param x: input data (3 x 6 x 7 tensor)
        :return: tuple consisting of computed policy and value for the input data.
        """
        x = F.relu(self.bn2d_1(self.conv1(x)))

        # Compute output of each layer (uses dropout of 0.3)
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=0.3, training=self.training)

        # Compute policy and value from residual block output
        return self.policy_head(x), self.value_head(x)

    def train_on_examples(self, examples, num_epochs=10, lr=0.001, weight_decay=0.0001, logger=None):
        """
        Train model on all data collected so far.
        Adapted from: https://github.com/suragnair/alpha-zero-general/tree/master
        :param examples: list containing all training data collected so far (unshuffled).
        :param num_epochs: number of epochs to train for.
        :param lr: learning rate.
        :param weight_decay: L2 penalty.
        :param logger: logger to record all results per iteration.
        :return: trained model.
        """

        # Shuffle examples to break correlations.
        shuffled_examples = copy.deepcopy(examples)
        np.random.shuffle(shuffled_examples)

        # Set up adam optimizer and model.
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.to(device)

        # Start training.
        for epoch in range(num_epochs):
            model.train()

            # Metrics to keep track of value and policy loss/accuracy.
            sum_total_loss = 0.0
            sum_value_loss = 0.0
            sum_policy_loss = 0.0
            sum_policy_acc = 0.0
            total_policy_acc = 0

            # Set up batches of examples and iterate over them.
            batch_size = 64
            batch_count = max(1, int(len(shuffled_examples) / batch_size))
            for i in range(batch_count):
                # Randomly selected batches of examples.
                batch = np.random.randint(len(shuffled_examples), size=batch_size)
                boards, move, value = list(zip(*[shuffled_examples[j] for j in batch]))
                boards = torch.tensor(np.array(boards)).float().to(device)

                target_move = torch.tensor(np.array(move)).float().to(device)
                target_value = torch.tensor(np.array(value)).float().to(device)

                # Compute value and policy.
                out_move, out_value = self(boards)
                loss_move = self.policy_loss(target_move, out_move)
                loss_value = self.value_loss(target_value, out_value)
                total_loss = loss_move + loss_value

                # Record results.
                sum_total_loss += total_loss.item()
                sum_policy_loss += loss_move.item()
                sum_value_loss += loss_value.item()
                total_policy_acc += out_move.size(0)
                sum_policy_acc += (torch.sum(torch.argmax(target_move, dim=1) == torch.argmax(out_move, dim=1))).item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Compute average across all batches.
            total_loss = sum_total_loss / batch_count
            value_loss = sum_value_loss / batch_count
            policy_acc = sum_policy_acc / total_policy_acc

            # Log results
            print(f"Epoch {epoch + 1}. "
                  f"Total loss: {total_loss:.3f}. "
                  f"Value loss: {value_loss:.3f}. "
                  f"Policy acc.: {policy_acc:.3f}")
            logger.log(f"(Training) Epoch: {epoch + 1}. Total loss: {total_loss:.3f}. Value loss: {value_loss:.3f}. "
                       f"Policy accuracy: {policy_acc:.3f}", to_summary=True, to_iteration=True)
        return copy.deepcopy(self)

    def policy_loss(self, targets, outputs):
        """
        Computes the cross-entropy loss between the specified policy targets and outputs. The outputs are not do not
        go through a logarithm as the policy network already outputs and log-softmax.
        Taken from: https://github.com/suragnair/alpha-zero-general/tree/master
        :param targets: policy targets (ground truth).
        :param outputs: policy outputs (model)
        :return: computed cross-entropy loss.
        """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def value_loss(self, targets, outputs):
        """
        Computes the mean-squared error loss between the specified policy targets and outputs.
        Taken from: https://github.com/suragnair/alpha-zero-general/tree/master
        :param targets: value targets (ground truth).
        :param outputs: value outputs (model).
        :return: computed mean-squared error loss.
        """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


class ResidualBlock(nn.Module):
    """
    Class to handle residual blocks.
    """
    def __init__(self, in_size, hidden_size, out_size, kernel_size=(3, 3), padding=0):
        """
        Constructor for the residual blocks.
        :param in_size: number of input filters.
        :param hidden_size: number of hidden filters (between the two convolutional layers).
        :param out_size: number of output filters.
        :param kernel_size: kernel size of each convolutional layer.
        :param padding: padding of each convolutional layer.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(hidden_size)
        self.bn2d_2 = nn.BatchNorm2d(out_size)

    def conv_block(self, x):
        """
        Sequence of two convolutional layers.
        :param x: input data.
        :return: data after being passed through all convolutional layers.
        """
        x = F.relu(self.bn2d_1(self.conv1(x)))
        x = F.relu(self.bn2d_2(self.conv2(x)))
        return x

    def forward(self, x):
        """
        Forward function for the residual block.
        :param x: input data.
        :return: sum of before and after passing the input through the two convolutional layers.
        """
        return x + self.conv_block(x)


class ValueHead(nn.Module):
    """
    Class to handle the value head of the residual network.
    """
    def __init__(self, in_size, fc_size, kernel_size=(1, 1), padding=0):
        """
        Constructor for the value head.
        :param in_size: number of input filters.
        :param fc_size: number of nodes in the only fully-connected layer.
        :param kernel_size: kernel size of the convolutional layer.
        :param padding: padding of the convolutional layer.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, 32, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(2304, fc_size)
        self.value = nn.Linear(fc_size, 1)

    def forward(self, x):
        """
        Forward function for the value head.
        :param x: input data.
        :return: output of the network passed through a tanh function ([-1, 1]).
        """
        x = F.relu(self.bn2d_1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.value(x))


class PolicyHead(nn.Module):
    """
    Class to handle the policy head.
    """
    def __init__(self, in_size, kernel_size=(1, 1), padding=0):
        """
        Constructor for the policy head.
        :param in_size: number of input filters.
        :param kernel_size: kernel size of the convolutional layer.
        :param padding: padding of the convolutional layer.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, 32, kernel_size=kernel_size, padding=padding)
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.policy = nn.Linear(2304, 7)

    def forward(self, x):
        """
        Forward function for the policy head.
        :param x: input data.
        :return: output of the network passed through a log-softmax function.
        """
        x = F.relu(self.bn2d_1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.policy(x)
        return F.log_softmax(x, dim=1)
