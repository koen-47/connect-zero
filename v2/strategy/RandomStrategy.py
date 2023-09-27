import random
from abc import ABC
import numpy as np

from v2.strategy.Strategy import Strategy


class RandomStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, board):
        valid_moves = board[0] == 0
        moves = [i for i, move in enumerate(valid_moves) if move]
        move = random.choice(moves)
        return move
