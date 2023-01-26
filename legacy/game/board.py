from typing import List
from termcolor import colored
import numpy as np


class Board:
    def __init__(self, num_rows: int, num_cols: int, board=None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.board: List[List[int]] = self.__create_board() if board is None else board
        self.is_game_over = False
        self.reward = {
            "win": 1,
            "draw": 0.5,
            "ongoing": 0,
            "lose": -1
        }

    def __create_board(self):
        return [[0] * self.num_cols for i in range(self.num_rows)]

    def drop(self, player: int, col: int):
        # if self.board[0][col] != 0:
        #     raise Exception("Invalid move.")

        i: int = self.num_rows
        for j in range(self.num_rows):
            if self.board[j][col] != 0:
                i = j
                break

        self.board[i - 1][col] = player

    def check_win(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (i + 3) < self.num_rows:
                    if self.board[i][j] == 1 and self.board[i + 1][j] == 1 and self.board[i + 2][j] == \
                            1 and self.board[i + 3][j] == 1:
                        return 1
                    elif self.board[i][j] == 2 and self.board[i + 1][j] == 2 and self.board[i + 2][j] == \
                            2 and self.board[i + 3][j] == 2:
                        return 2

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (j + 3) < self.num_cols:
                    if self.board[i][j] == 1 and self.board[i][j + 1] == 1 and self.board[i][j + 2] == \
                            1 and self.board[i][j + 3] == 1:
                        return 1
                    if self.board[i][j] == 2 and self.board[i][j + 1] == 2 and self.board[i][j + 2] == \
                            2 and self.board[i][j + 3] == 2:
                        return 2

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (i - 3) < self.num_rows and (j + 3) < self.num_cols:
                    if self.board[i][j] == 1 and self.board[i - 1][j + 1] == 1 and self.board[i - 2] \
                            [j + 2] == 1 and self.board[i - 3][j + 3] == 1:
                        return 1
                    if self.board[i][j] == 2 and self.board[i - 1][j + 1] == 2 and self.board[i - 2] \
                            [j + 2] == 2 and self.board[i - 3][j + 3] == 2:
                        return 2

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (i - 3) < self.num_rows and (j - 3) < self.num_cols:
                    if self.board[i][j] == 1 and self.board[i - 1][j - 1] == 1 and \
                            self.board[i - 2][j - 2] == 1 and self.board[i - 3][j - 3] == 1:
                        return 1
                    if self.board[i][j] == 2 and self.board[i - 1][j - 1] == 2 and \
                            self.board[i - 2][j - 2] == 2 and self.board[i - 3][j - 3] == 2:
                        return 2

        if all(cell != 0 for cell in self.board[0]):
            return 0
        return -1

    def get_valid_moves(self):
        valid_moves = []
        for i in range(len(self.board)):
            if self.board[0][i] == 0:
                valid_moves.append(i)
        return valid_moves

    def get_board(self):
        return self.board

    def get_num_rows(self):
        return self.num_rows

    def get_num_cols(self):
        return self.num_cols

    def clone(self):
        new_board = Board(self.num_rows, self.num_cols, board=self.board)
        return new_board

    def calculate_rewards(self, player_id: int):
        winner = self.check_win()
        if winner == -1:
            return self.reward["ongoing"]

        if winner == 1 and player_id == 1:
            return self.reward["win"]
        elif winner == 1 and player_id == 2:
            return self.reward["lose"]
        elif winner == 2 and player_id == 1:
            return self.reward["lose"]
        elif winner == 2 and player_id == 2:
            return self.reward["win"]
        return self.reward["draw"]

    def get_board_copy(self):
        return self.board.copy()

    def set_board(self, board: List[List[int]]):
        self.board = board
        self.num_rows = len(board)
        self.num_cols = len(board[0])

    def to_string(self):
        s: str = ""
        for row in self.board:
            for cell in row:
                if cell == 1:
                    s += "X "
                elif cell == 2:
                    s += "O "
                else:
                    s += "- "
            s += "\n"
        return s
