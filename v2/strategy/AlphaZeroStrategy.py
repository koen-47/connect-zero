import torch
from abc import ABC

# from v2.strategy.Strategy import Strategy


class AlphaZeroStrategy:
    def __init__(self, mcts):
        self.mcts = mcts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_move(self, board, player_id):
        action, probs = self.mcts.get_action_prob(board, device=self.device)
        return action, probs
