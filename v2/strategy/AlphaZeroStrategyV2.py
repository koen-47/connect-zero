import copy
import logging
import math
from abc import ABC

import numpy as np
import torch
from collections import defaultdict

EPS = 1e-8


class AlphaZeroStrategyV2:
    def __init__(self, mcts):
        self.mcts = mcts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MCTS:
    def __init__(self, game, nnet, device, num_sims=250, c_puct=1.):
        self.game = game
        self.nnet = nnet.to(device)
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.nodes = {}

        self.temp = 1
        self.dir_alpha = 1.
        self.dir_e = 0.25

    def get_action_prob(self, canonical_board, device):
        for i in range(self.num_sims):
            self.search(canonical_board, is_root=True, device=device)

        node = self.nodes[tuple(map(tuple, canonical_board))]
        counts = [node.child_visits[a] if node.child_visits[a] != 0 else 0 for a in range(self.game.get_action_size())]

        if self.temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / self.temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board, is_root, device):
        node_key = tuple(map(tuple, board))
        if node_key not in self.nodes:
            self.nodes[node_key] = Node(board)
        node = self.nodes[node_key]

        if node.reward is None:
            node.reward = self.game.get_game_ended(board, 1)
        if node.reward != 0:
            return -node.reward

        if node.policy is None:
            tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
            tensor_state = tensor_state.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
            policy, value = self.nnet(tensor_state)
            policy = policy.detach().cpu().numpy().flatten()
            value = value.detach().cpu().numpy().flatten()[0]
            valids = self.game.get_valid_moves(board)
            policy = policy * valids
            node.policy = policy
            node.valid_moves = valids
            node.n_visits = 0
            return -value

        valid_moves = node.valid_moves
        cur_best = -float('inf')
        best_act = -1

        if is_root and self.dir_e > 0:
            noise = np.random.dirichlet([self.dir_alpha] * len(valid_moves))

        i = 0
        for a in range(self.game.get_action_size()):
            if valid_moves[a]:
                p = node.policy[a]
                if is_root and self.dir_e > 0:
                    p = (1 - self.dir_e) * p + self.dir_e * noise[i]
                i += 1

                if node.q_values[a] is not None:
                    u = node.q_values[a] + self.c_puct * p * math.sqrt(node.n_visits) / (1 + node.child_visits[a])
                else:
                    u = self.c_puct * p * math.sqrt(node.n_visits + EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)
        v = self.search(next_s, is_root=False, device=device)

        if node.q_values[a] is not None:
            node.q_values[a] = (node.child_visits[a] * node.q_values[a] + v) / (node.child_visits[a] + 1)
            node.child_visits[a] += 1
        else:
            node.q_values[a] = v
            node.child_visits[a] = 1

        node.n_visits += 1
        return -v


class Node:
    def __init__(self, state, reward=None, valid_moves=None, policy=None):
        if valid_moves is None:
            valid_moves = []
        self.state = state
        self.reward = reward
        self.valid_moves = valid_moves
        self.policy = policy

        self.q_values = [None] * 7
        self.child_visits = [0] * 7
        self.n_visits = 0
        self.actions = []

    def is_terminal(self):
        return self.reward != 0

    def increment_visit(self):
        self.n_visits += 1
