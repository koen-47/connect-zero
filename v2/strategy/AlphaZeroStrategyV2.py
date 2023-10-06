import torch
from abc import ABC

from v2.strategy.Strategy import Strategy


class AlphaZeroStrategyV2(ABC, Strategy):
    def __init__(self, mcts):
        super().__init__()
        self.mcts = mcts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_move(self, board, player_id):
        best_action, probs = self.mcts.get_action_prob(board, device=self.device)
        # print(probs)
        return best_action
