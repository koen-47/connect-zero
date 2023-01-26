from strategies.strategy import RandomStrategy, ManualStrategy, RLStrategy, AlphaBetaPruningStrategy


class Player:
    def __init__(self, id: int, strategy: str = "random"):
        self.id = id
        if strategy == "random":
            self.strategy = RandomStrategy()
        elif strategy == "manual":
            self.strategy = ManualStrategy()
        elif strategy == "rl":
            self.strategy = RLStrategy()
        elif strategy == "alpha_beta":
            self.strategy = AlphaBetaPruningStrategy()

    def move(self, board):
        col = self.strategy.calculate_move(board.board)
        board.drop(self.id, col)

