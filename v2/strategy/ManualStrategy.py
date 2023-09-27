from abc import ABC

from v2.strategy.Strategy import Strategy


class ManualStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, board):
        return int(input("Enter a column to drop: ")) - 1

