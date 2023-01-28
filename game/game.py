from .board import Board
from .player import Player


class Game:
    def __init__(self):
        self.board: Board = Board(6, 7)

    def play_manual(self):
        turn = 1
        player = Player(1, strategy="manual")
        cpu = Player(2, strategy="classification")
        while not self.is_game_over():
            print(f"Turn {turn}")
            player.move(self.board)
            if self.board.check_win() == 1:
                break
            cpu.move(self.board)
            print(self.board.to_string())
            turn += 1

        winner = self.board.check_win()
        if winner == 1:
            print("Player 1 wins!")
        elif winner == 2:
            print("Player 2 wins!")
        elif winner == 0:
            print("Draw!")
        print(self.board.to_string())

    def is_game_over(self):
        win_condition = self.board.check_win()
        self.board.is_game_over = win_condition
        return win_condition != -1

    def get_board_copy(self):
        return self.board.board.copy()

    def reset(self):
        self.__init__()
