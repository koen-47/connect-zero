import copy
import math
import random

import numpy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import count

from models.cnn.dqn import DQN

from game.game import Game
from board import Board


class replayMemory:
    def __init__(self):
        self.memory = []

    def dump(self, transition_tuple):
        self.memory.append(transition_tuple)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = replayMemory()


class CNNStrategy:
    def __init__(self):
        self.BATCH_SIZE = 256
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.5
        self.EPS_DECAY = 2000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.n_actions = 7
        self.height = 6
        self.width = 7

        self.policy_net = DQN(self.n_actions).to(self.device)
        self.target_net = DQN(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.game = Game()
        self.memory = replayMemory()

    def select_action(self, board: Board, available_actions, steps_done=None, training=True):
        state = torch.tensor(board.board, dtype=torch.float, device=self.device).unsqueeze(dim=0).unsqueeze(dim=0)
        epsilon = random.random()
        if training:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * steps_done / self.EPS_DECAY)
        else:
            eps_threshold = 0

        # follow epsilon-greedy policy
        if epsilon > eps_threshold:
            with torch.no_grad():
                # action recommendations from policy net
                r_actions = self.policy_net(state)[0, :].to("cpu")
                state_action_values = [r_actions[action] for action in available_actions]
                argmax_action = np.argmax(state_action_values)
                greedy_action = available_actions[argmax_action]
                return greedy_action
        else:
            return random.choice(available_actions)

    def optimize_model(self):
        if len(memory) < self.BATCH_SIZE:
            return
        transitions = memory.sample(self.BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), \
                                                                           [m[1]], m[2], np.expand_dims(m[3], axis=0))
                                                                          for m in transitions])
        # tensor wrapper
        # state_batch = [batch.board for batch in state_batch]
        # print(state_batch)
        state_batch = torch.tensor(numpy.array(state_batch), dtype=torch.float, device=self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=self.device)

        # print(next_state_batch)
        # for assigning terminal state value = 0 later
        non_final_mask = torch.tensor(tuple(map(lambda s_: s_[0] is not None, next_state_batch)), device=self.device)
        non_final_next_state = torch.cat(
            [torch.tensor(s_, dtype=torch.float, device=self.device).unsqueeze(0) for s_ in next_state_batch if
             s_[0] is not None])

        # prediction from policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # truth from target_net, initialize with zeros since terminal state value = 0
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad
        next_state_values[non_final_mask] = self.target_net(non_final_next_state).max(1)[0].detach()
        # compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))  # torch.tensor.unsqueeze returns a copy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def random_agent(self, actions):
        return random.choice(actions)

    # win rate test
    def win_rate_test(self):
        win_moves_taken_list = []
        win = []
        for i in range(100):
            self.game.reset()
            win_moves_taken = 0

            while not self.game.is_game_over():
                board_copy = copy.copy(self.game.board)
                available_actions = board_copy.get_valid_moves()
                action = self.select_action(board_copy, available_actions, training=False)
                board_copy.drop(1, action)
                reward = board_copy.calculate_rewards(1)
                win_moves_taken += 1

                if reward == 1:
                    win_moves_taken_list.append(win_moves_taken)
                    win.append(1)
                    break

                available_actions = board_copy.get_valid_moves()
                action = self.random_agent(available_actions)
                board_copy.drop(2, action)
                reward = board_copy.calculate_rewards(2)
                self.game.board = board_copy

        return sum(win) / 100, sum(win_moves_taken_list) / len(win_moves_taken_list)

    def train(self):
        steps_done = 0
        training_history = []

        num_episodes = 100
        # control how lagged is target network by updating every n episodes
        TARGET_UPDATE = 10

        for i in range(num_episodes):
            print(i)
            self.game.reset()
            state_p1 = copy.copy(self.game.board)

            # record every 20 epochs
            if i % 20 == 0:
                win_rate, moves_taken = self.win_rate_test()
                training_history.append([i + 1, win_rate, moves_taken])
                th = np.array(training_history)
                # print training message every 200 epochs
                # if i % 10 == 0:
                print('Episode {}: | win_rate: {} | moves_taken: {}'.format(i, th[-1, 1], th[-1, 2]))

            for t in count():
                available_actions = self.game.board.get_valid_moves()
                action_p1 = self.select_action(state_p1, available_actions, steps_done)
                steps_done += 1
                # print(action_p1)
                # print(self.game.board.to_string())
                state_p1.drop(1, action_p1)
                state_p1_ = state_p1
                reward_p1 = state_p1_.calculate_rewards(1)

                if state_p1.is_game_over:
                    if reward_p1 == 1:
                        # reward p1 for p1's win
                        memory.dump([state_p1.board, action_p1, 1, None])
                    else:
                        # state action value tuple for a draw
                        memory.dump([state_p1.board, action_p1, 0.5, None])
                    break

                available_actions = state_p1_.get_valid_moves()
                if len(available_actions) > 0:
                    # print(available_actions)
                    action_p2 = self.random_agent(available_actions)
                    state_p1_.drop(2, action_p2)

                state_p2_ = state_p1_
                if len(available_actions) == 0:
                    state_p2_.is_game_over = True
                reward_p2 = state_p2_.calculate_rewards(2)

                if state_p2_.is_game_over:
                    if reward_p2 == 1:
                        # punish p1 for (random agent) p2's win
                        memory.dump([state_p1.board, action_p1, -1, None])
                    else:
                        # state action value tuple for a draw
                        memory.dump([state_p1.board, action_p1, 0.5, None])
                    break

                # punish for taking too long to win
                memory.dump([state_p1.board, action_p1, -0.05, state_p2_.board])
                state_p1 = state_p2_

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

            # update the target network, copying all weights and biases in DQN
            if i % TARGET_UPDATE == TARGET_UPDATE - 1:
                self.target_net.load_state_dict(self.policy_net.state_dict())


strat = CNNStrategy()
strat.train()
