class Player:
    def __init__(self, id, strategy):
        self.id = id
        self.strategy = strategy

    def clone(self):
        player = Player(self.id, self.strategy)
        return player
