from strategy import RandomStrategy, ManualStrategy


class Player:
    def __init__(self, id: int, strategy: str = "random"):
        self.id = id
        if strategy == "random":
            self.strategy = RandomStrategy()
        elif strategy == "manual":
            self.strategy = ManualStrategy()

    def move(self, board):
        try:
            col = self.strategy.calculate_move(board)
            board.drop(self.id, col)
        except Exception:
            # print("Invalid move! Trying again...")
            col = self.strategy.calculate_move(board)
            board.drop(self.id, col)
