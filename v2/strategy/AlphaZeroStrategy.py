import copy
import logging
import math
from abc import ABC

import numpy as np
import torch

from v2.strategy.Strategy import Strategy

EPS = 1e-8

log = logging.getLogger(__name__)


class AlphaZeroStrategy:
    def __init__(self, mcts):
        self.mcts = mcts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_move(self, canonical_board, player_id):
        pi = self.mcts.get_action_prob(canonical_board, device=self.device, temp=0)
        # print(pi)
        # print(game.get_valid_moves(canonicalBoard).astype(int))
        # print(pi * game.get_valid_moves(canonicalBoard).astype(int))
        action = np.random.choice(len(pi), p=pi)
        return action


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, device, num_sims=250, cpuct=1.0):
        self.game = game
        self.nnet = nnet.to(device)
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def get_action_prob(self, canonical_board, device, temp=1, dir_alpha=1., dir_e=0.25):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.num_sims):
            self.search(canonical_board, is_root=True, device=device, dir_alpha=dir_alpha, dir_e=dir_e)

        s = self.game.get_string_representation(canonical_board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        # print(s)
        # print([n for n in counts])

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonical_board, is_root, device, dir_alpha=1., dir_e=0.25):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.get_string_representation(canonical_board)

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        # print(s)
        if s not in self.Ps:
            tensor_state = torch.tensor(np.array(canonical_board), dtype=torch.float32).unsqueeze(dim=0).unsqueeze(
                dim=0).to(device)
            self.Ps[s], value = self.nnet(tensor_state)
            self.Ps[s] = self.Ps[s].detach().cpu().numpy().flatten()
            v = value.detach().cpu().numpy().flatten()[0]
            valids = self.game.get_valid_moves(canonical_board)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        if is_root and dir_e > 0:
            noise = np.random.dirichlet([dir_alpha] * len(valids))

        i = -1
        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valids[a]:
                i += 1
                p = self.Ps[s][a]
                if is_root and dir_e > 0:
                    p = (1 - dir_e) * p + dir_e * noise[i]

                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * p * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * p * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s, is_root=False, device=device, dir_alpha=dir_alpha, dir_e=dir_e)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
