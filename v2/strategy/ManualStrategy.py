from abc import ABC

from v2.strategy.Strategy import Strategy


class ManualStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, game, player_id):
        return int(input("Enter a column to drop: ")) - 1

