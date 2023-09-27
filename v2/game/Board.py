import copy

import numpy as np


class Board:
    def __init__(self, height=6, width=7, state=None):
        self.state = np.zeros([height, width], dtype=int) if state is None else state
        self.height = height
        self.width = width

    def move(self, player_id, col):
        for j in reversed(range(self.height)):
            if self.state[j][col] == 0:
                self.state[j][col] = player_id
                break

    def get_valid_moves(self):
        return self.state[0] == 0

    def get_status(self):
        if sum(self.get_valid_moves()) == 0:
            return None

        for i in range(self.height):
            for j in range(self.width):
                if (i + 3) < self.height:
                    if self.state[i][j] == 1 and self.state[i + 1][j] == 1 and self.state[i + 2][j] == \
                            1 and self.state[i + 3][j] == 1:
                        return 1
                    elif self.state[i][j] == -1 and self.state[i + 1][j] == -1 and self.state[i + 2][j] == \
                            -1 and self.state[i + 3][j] == -1:
                        return -1

        for i in range(self.height):
            for j in range(self.width - 3):
                if self.state[i][j] == 1 and self.state[i][j + 1] == 1 and self.state[i][j + 2] == \
                        1 and self.state[i][j + 3] == 1:
                    return 1
                if self.state[i][j] == -1 and self.state[i][j + 1] == -1 and self.state[i][j + 2] == \
                        -1 and self.state[i][j + 3] == -1:
                    return -1

        for i in range(self.height - 3):
            for j in range(self.width - 4, self.width):
                if self.state[i][j] == 1 and self.state[i + 1][j - 1] == 1 and self.state[i + 2] \
                        [j - 2] == 1 and self.state[i + 3][j - 3] == 1:
                    return 1
                if self.state[i][j] == -1 and self.state[i + 1][j - 1] == -1 and self.state[i + 2] \
                        [j - 2] == -1 and self.state[i + 3][j - 3] == -1:
                    return -1

        for i in range(self.height - 3):
            for j in range(self.width - 3):
                if self.state[i][j] == 1 and self.state[i + 1][j + 1] == 1 and \
                        self.state[i + 2][j + 2] == 1 and self.state[i + 3][j + 3] == 1:
                    return 1
                if self.state[i][j] == -1 and self.state[i + 1][j + 1] == -1 and \
                        self.state[i + 2][j + 2] == -1 and self.state[i + 3][j + 3] == -1:
                    return -1
        return 0

    def clone(self, state):
        return copy.deepcopy(Board(height=self.height, width=self.width, state=state))

    def get_string_representation(self):
        return str(self.state)

    def __str__(self):
        board_str = ""
        for i in range(self.height):
            for j in range(self.width):
                cell_str = "- "
                if self.state[i][j] == 1:
                    cell_str = "X "
                elif self.state[i][j] == -1:
                    cell_str = "O "
                board_str += cell_str
            board_str += "\n"
        return board_str
