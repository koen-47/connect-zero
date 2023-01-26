from typing import List
from termcolor import colored
import numpy as np


class Board:
    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.board: List[List[int]] = self.__create_board()
        self.is_game_over = False
        self.reward = {
            "win": 1,
            "draw": 0.5,
            "lose": -1
        }

    def __create_board(self):
        return [[0] * self.num_cols for i in range(self.num_rows)]

    def drop(self, player: int, col: int):
        if self.board[0][col] != 0:
            raise Exception("Invalid move.")

        i: int = self.num_rows
        for j in range(self.num_rows):
            if self.board[j][col] != 0:
                i = j
                break

        self.board[i - 1][col] = player

    def check_win(self, player_id: int):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (i + 3) < self.num_rows:
                    if self.board[i][j] == player_id and self.board[i + 1][j] == player_id and self.board[i + 2][j] == \
                            player_id and self.board[i + 3][j] == player_id:
                        self.is_game_over = True

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (j + 3) < self.num_cols:
                    if self.board[i][j] == player_id and self.board[i][j + 1] == player_id and self.board[i][j + 2] == \
                            player_id and self.board[i][j + 3] == player_id:
                        self.is_game_over = True

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (i - 3) < self.num_rows and (j + 3) < self.num_cols:
                    if self.board[i][j] == player_id and self.board[i - 1][j + 1] == player_id and self.board[i - 2] \
                            [j + 2] == player_id and self.board[i - 3][j + 3] == player_id:
                        self.is_game_over = True

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if (i - 3) < self.num_rows and (j - 3) < self.num_cols:
                    if self.board[i][j] == player_id and self.board[i - 1][j - 1] == player_id and \
                            self.board[i - 2][j - 2] == player_id and self.board[i - 3][j - 3] == player_id:
                        self.is_game_over = True

        if self.is_game_over is True:
            return self.reward["win"]
        if all(all(cell != 0 for cell in row) for row in self.board):
            print(all(all(cell != 0 for cell in row) != 0 for row in self.board))
            self.is_game_over = True
            return self.reward["draw"]
        return 0

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

    def calculate_rewards(self):
        pass

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
