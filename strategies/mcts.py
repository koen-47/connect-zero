import copy
from typing import List

import numpy as np
import torch
import math

from brain.classification.train_classifier import Classifier1
from game.board import Board
from game.game import Game
import game.util as util


class MCTS:
    """
    Code for this class was adapted from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
    """

    def __init__(self, game, model, player_id, num_sims=25, cpuct=1):
        self.game = game
        self.model = Classifier1()
        self.player_id = player_id
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def get_action_probability(self, board: List[List[int]], player_id: int, temp=1):
        for i in range(self.num_sims):
            self.search(copy.deepcopy(board), player_id)

        state = np.array2string(np.array(board))
        # print(state)
        # print(self.Nsa)
        counts = [self.Nsa[(state, action)] if (state, action) in self.Nsa else 0 for action in range(7)]

        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            print(probs)
            return probs

        # print(counts)
        # print(self.Qsa)
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        print(probs)
        return probs

    def search(self, board: List[List[int]], next_player_id: int):
        state = np.array2string(np.array(board))

        if state not in self.Es:
            self.Es[state] = util.check_win(board)
        if self.Es[state] != -1:
            if self.Es[state] == 1:
                return 2
            if self.Es[state] == 2:
                return 1
            return 0

        if state not in self.Ps:
            tensor_state = torch.tensor(np.array(board), dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)
            self.Ps[state], value = self.model(tensor_state)
            self.Ps[state] = self.Ps[state].detach().numpy().flatten()
            value = value.detach().numpy().flatten()[0]
            # print(value)
            valid_moves = np.where(np.array(board[0]) > 0, 0, 1)
            self.Ps[state] = self.Ps[state] * valid_moves
            sum_Ps_s = np.sum(self.Ps[state])
            if sum_Ps_s > 0:
                self.Ps[state] /= sum_Ps_s
            else:
                self.Ps[state] = self.Ps[state] + valid_moves
                self.Ps[state] /= np.sum(self.Ps[state])

            self.Vs[state] = valid_moves
            self.Ns[state] = 0
            return -value

        valid_moves = self.Vs[state]
        current_best = -float("inf")
        best_action = -1

        for action in range(7):
            if valid_moves[action]:
                if (state, action) in self.Qsa:
                    u = self.Qsa[(state, action)] + self.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state]) \
                        / (1 + self.Nsa[(state, action)])
                else:
                    # print(self.Ps[state])
                    u = self.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state] + 1e-8)
                    # print(u)

                # print(u)
                # print(current_best)
                if u > current_best:
                    current_best = u
                    best_action = action

        action = best_action
        next_state = util.drop(board, 1, action)
        # next_player_id = 2 if next_player_id == 1 else 1

        # print(np.array(board.board))

        value = self.search(next_state, 1)

        if (state, action) in self.Qsa:
            self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + value) / \
                                        (self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1
        else:
            self.Qsa[(state, action)] = value
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1
        return -value
