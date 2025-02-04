import random
from abc import ABC
import numpy as np

from strategy.Strategy import Strategy


class RandomStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, board, player_id):
        moves = [i for i, move in enumerate(board[0] == 0) if move]
        move = random.choice(moves)
        policy = np.zeros(len(board[0]))
        policy[move] = 1
        # print(policy)
        return move, policy
