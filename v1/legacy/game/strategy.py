import random
from abc import ABC, abstractmethod

from board import Board


class Strategy:
    def __init__(self):
        pass

    @abstractmethod
    def calculate_move(self, board):
        pass


class RandomStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, board: Board):
        moves = board.get_valid_moves()
        return random.choice(moves)


class ManualStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, board: Board):
        return int(input("Enter a column to drop: "))


