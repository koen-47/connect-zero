import torch


class AlphaZeroStrategyV2:
    def __init__(self, mcts):
        self.mcts = mcts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_move(self, board):
        best_action, _ = self.mcts.get_action_prob(board, device=self.device)
        return best_action
