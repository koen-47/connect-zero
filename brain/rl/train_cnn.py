import json
import logging
import math
import random
from typing import List

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import count

from models.cnn.cnn_dqn_1 import CNN_DQN_1
from models.cnn.cnn_dqn_2 import CNN_DQN_2

import game.game as game
from replay_memory import ReplayMemory
import rewards
from strategies.strategy import AlphaBetaPruningStrategy, RandomStrategy


class CNNStrategy:
    def __init__(self, policy_net=None):
        self.BATCH_SIZE = 256
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.5
        self.EPS_DECAY = 2000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.n_actions = 7
        self.height = 6
        self.width = 7

        self.policy_net = CNN_DQN_2(self.n_actions).to(self.device) if policy_net is None else policy_net
        self.target_net = CNN_DQN_2(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.game = game.Game()
        self.memory = ReplayMemory()

        self.opponent_strat = AlphaBetaPruningStrategy(depth=0)

    def select_action(self, board: List[List[int]], available_actions, steps_done=None, training=True):
        state = torch.tensor(board, dtype=torch.float, device=self.device).unsqueeze(dim=0).unsqueeze(dim=0)
        epsilon = random.random()
        if training:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * steps_done / self.EPS_DECAY)
        else:
            eps_threshold = 0

        if epsilon > eps_threshold:
            with torch.no_grad():
                r_actions = self.policy_net(state)[0, :].to("cpu")
                state_action_values = [r_actions[action] for action in available_actions]
                argmax_action = np.argmax(state_action_values)
                greedy_action = available_actions[argmax_action]
                return greedy_action
        else:
            return random.choice(available_actions)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*[(np.expand_dims(m[0], axis=0), [m[1]], m[2],
                                                                           np.expand_dims(m[3], axis=0))
                                                                          for m in transitions])
        # tensor wrapper
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float, device=self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=self.device)

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

    def win_rate_test(self):
        win_moves_taken_list = []
        win = []
        for i in range(100):
            self.game.reset()
            win_moves_taken = 0

            while not self.game.is_game_over():
                state = self.game.get_board_copy()
                available_actions = self.game.board.get_valid_moves()
                action = self.select_action(state, available_actions, training=False)
                state, reward = self.game.board.drop(1, action)
                win_moves_taken += 1

                if reward == 1:
                    win_moves_taken_list.append(win_moves_taken)
                    win.append(1)
                    break

                action = self.opponent_strat.calculate_move(state)
                state, reward = self.game.board.drop(2, action)

        self.game.reset()
        num_moves_taken = len(win_moves_taken_list) if len(win_moves_taken_list) > 0 else 1
        return sum(win) / 100, sum(win_moves_taken_list) / num_moves_taken

    def train(self):
        steps_done = 0
        training_history = []

        num_episodes = 20000
        # control how lagged is target network by updating every n episodes
        TARGET_UPDATE = 10

        logging.basicConfig(filename="training_info_history_2.log",
                            filemode='w+',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        for i in range(1, num_episodes+1):
            self.game.reset()
            state_p1 = self.game.get_board_copy()

            if i % 100 == 0:
                win_rate, moves_taken = self.win_rate_test()
                print(f'ep. {i}, win_rate: {win_rate}, moves_taken: {moves_taken}')
                torch.save(strategy.policy_net.state_dict(), "../../models/saved/dqn_cnn_11.pth")

            action_history = []
            for idx, t in enumerate(count()):
                available_actions = self.game.board.get_valid_moves()
                action_p1 = self.select_action(state_p1, available_actions, steps_done)
                steps_done += 1
                self.game.board.drop(1, action_p1)
                state_p1_ = self.game.board.board
                action_history.append((10 + action_p1,))

                game_status = self.game.board.check_win()
                if game_status == 1:
                    reward = rewards.constant_reward(1, game_status)
                    info = f"ep. {i}, game_status: {game_status}, reward p1: {reward}, action_history: {action_history}, {idx}"
                    board = np.array2string(np.array(self.game.board.board), separator=", ")
                    logging.debug(info)
                    logging.debug(f"\n{board}")
                    self.memory.dump([state_p1, action_p1, 1, None])
                    break
                elif game_status == 0:
                    self.memory.dump([state_p1, action_p1, 0.5, None])
                    break

                action_p2 = self.opponent_strat.calculate_move(state_p1_)
                self.game.board.drop(2, action_p2)
                state_p2_ = self.game.board.board
                action_history[-1] += (20 + action_p2,)

                game_status = self.game.board.check_win()
                if game_status == 2:
                    reward = rewards.constant_reward(1, game_status)
                    info = f"ep. {i}, game_status: {game_status}, reward p1: {reward}, action_history: {action_history}, {idx}"
                    board = np.array2string(np.array(self.game.board.board), separator=", ")
                    logging.debug(info)
                    logging.debug(f"\n{board}")
                    self.memory.dump([state_p1, action_p1, -1, None])
                    break
                elif game_status == 0:
                    self.memory.dump([state_p1, action_p1, 0.5, None])
                    break

                self.memory.dump([state_p1, action_p1, -0.05, state_p2_])
                state_p1 = state_p2_

                self.optimize_model()

            if i % TARGET_UPDATE == TARGET_UPDATE - 1:
                self.target_net.load_state_dict(self.policy_net.state_dict())


strategy = CNNStrategy()
strategy.train()
