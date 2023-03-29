import copy
from typing import List

import numpy as np
import torch
import math

import game.util as util


class MCTS:
    """
    Code for this class was adapted from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
    """

    def __init__(self, model, player_id, num_sims=25, cpuct=4.):
        self.model = model
        self.player_id = player_id
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def get_action_probability(self, board: List[List[int]], player_id: int, e: float, temp=1, device="cuda"):
        for i in range(self.num_sims):
            self.search(copy.deepcopy(board), player_id, is_root=True, e=e, device=device)

        state = np.array2string(np.array(board))
        # print(state)
        # print(self.Nsa)
        counts = [self.Nsa[(state, action)] if (state, action) in self.Nsa else 0 for action in range(7)]

        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            # print(f"{player_id}: {probs}")
            return probs

        # print(counts)
        # print(self.Qsa)
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board: List[List[int]], next_player_id: int, is_root: bool, e: float, device):
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
            tensor_state = tensor_state.to(device)
            self.Ps[state], value = self.model(tensor_state)
            self.Ps[state] = self.Ps[state].detach().cpu().numpy().flatten()
            # print(self.Ps[state])
            value = value.detach().cpu().numpy().flatten()[0]
            # print(value)
            valid_moves = np.where(np.array(board[0]) > 0, 0, 1)
            self.Ps[state] = self.Ps[state] * valid_moves
            sum_Ps_s = np.sum(self.Ps[state])
            if sum_Ps_s > 0:
                self.Ps[state] /= sum_Ps_s
            else:
                # print("asdf")
                self.Ps[state] = self.Ps[state] + valid_moves
                self.Ps[state] /= np.sum(self.Ps[state])

            self.Vs[state] = valid_moves
            self.Ns[state] = 0
            # print("leaf node encountered")
            return -value

        valid_moves = self.Vs[state]
        current_best = -float("inf")
        best_action = -1

        if is_root and e > 0:
            noise = np.random.dirichlet([1.] * len(valid_moves))

        i = -1
        for action in range(7):
            if valid_moves[action]:
                i += 1
                if (state, action) in self.Qsa:
                    p = self.Ps[state][action]
                    if is_root and e > 0:
                        p = (1 - e) * p + e * noise[i]
                    u = self.Qsa[(state, action)] + self.cpuct * p * math.sqrt(self.Ns[state]) \
                        / (1 + self.Nsa[(state, action)])
                else:
                    # print(self.Ps[state])
                    p = self.Ps[state][action]
                    if is_root and e > 0:
                        p = (1 - e) * p + e * noise[i]
                    u = self.cpuct * p * math.sqrt(self.Ns[state] + 1e-8)
                    # print(u)

                # print(u)
                # print(current_best)
                if u > current_best:
                    current_best = u
                    best_action = action

        action = best_action
        next_state = util.drop(board, next_player_id, action)
        next_player_id = 2 if next_player_id == 1 else 1

        # print(np.array(board))

        value = self.search(next_state, next_player_id, is_root=False, e=e, device=device)

        if (state, action) in self.Qsa:
            self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + value) / \
                                        (self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1
        else:
            self.Qsa[(state, action)] = value
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1
        return -value


# def execute_episode_mcts(num_games: int, num_sims: int = 25):
#     game = Game()
#     wins = [0, 0, 0]
#     examples = []
#     for i in range(num_games):
#         game.reset()
#
#         # if i % 100 == 0:
#         #     print(i)
#
#         turn_num = 1
#         p1_strategy = MCTS(model=None, player_id=1, num_sims=num_sims)
#         p2_strategy = MCTS(model=None, player_id=2, num_sims=num_sims)
#         local_training_data = []
#         while not game.is_game_over():
#             p1_move_enc = p1_strategy.get_action_probability(game.board.board, 1, temp=0)
#             p1_state = copy.deepcopy(game.board.board)
#             game.board.drop(1, np.argmax(p1_move_enc))
#             local_training_data.append((p1_state, p1_move_enc))
#
#             game_status = game.board.check_win()
#             if game_status != -1:
#                 break
#
#             p2_move_enc = p2_strategy.get_action_probability(game.board.board, 2, temp=0)
#             p2_state = copy.deepcopy(game.board.board)
#             game.board.drop(2, np.argmax(p2_move_enc))
#             local_training_data.append((p2_state, p2_move_enc))
#             turn_num += 1
#
#         game_status = game.board.check_win()
#         if game_status == 2:
#             wins[2] += 1
#         elif game_status == 0:
#             wins[1] += 1
#         elif game_status == 1:
#             wins[0] += 1
#         for data in local_training_data:
#             data = data + (-1 if game_status == 2 else game_status,)
#             examples.append(data)
#         # print(np.array(game.board.board))
#
#     print(wins)
#     return examples
