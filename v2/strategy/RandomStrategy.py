import random
from abc import ABC
import numpy as np

from v2.strategy.Strategy import Strategy


class RandomStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, game, player_id):
        move = random.choice([i for i, move in enumerate(game.get_valid_moves(game.board.state)) if move == True])
        return move