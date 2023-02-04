from strategies.strategy import RandomStrategy, ManualStrategy, RLStrategy, AlphaBetaPruningStrategy, ClassificationStrategy


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
            perspective = 1 if id == 2 else 2
            self.strategy = AlphaBetaPruningStrategy(player_id=perspective, depth=3)
        elif strategy == "classification":
            self.strategy = ClassificationStrategy()

    def move(self, board):
        col = self.strategy.calculate_move(board.board)
        print(f"player {self.id}, col: {col}")
        board.drop(self.id, col)

