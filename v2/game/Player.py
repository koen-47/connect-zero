from v2.strategy.RandomStrategy import RandomStrategy
from v2.strategy.ManualStrategy import ManualStrategy
from v2.strategy.AlphaZeroStrategy import AlphaZeroStrategy


class Player:
    def __init__(self, id, strategy):
        self.id = id
        self.strategy = strategy

    def clone(self):
        player = Player(self.id, self.strategy)
        return player
