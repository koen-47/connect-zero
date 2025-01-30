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

        if sum(self.get_valid_moves()) == 0:
            return 1e-4

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


def encode_board(board, player):
    if player == 1:
        plane1 = (board == 1).astype(np.float32).tolist()
        plane2 = (board == -1).astype(np.float32).tolist()
        plane3 = np.ones_like(board, dtype=np.float32).tolist()
    else:
        plane1 = (board == -1).astype(np.float32).tolist()
        plane2 = (board == 1).astype(np.float32).tolist()
        plane3 = np.zeros_like(board, dtype=np.float32).tolist()

    planes = np.stack([plane1, plane2, plane3], axis=0).tolist()
    return list(planes)


def get_status(state):
    height, width = 6, 7
    for i in range(height):
        for j in range(width):
            if (i + 3) < height:
                if state[i][j] == 1 and state[i + 1][j] == 1 and state[i + 2][j] == 1 and state[i + 3][j] == 1:
                    return 1
                elif state[i][j] == -1 and state[i + 1][j] == -1 and state[i + 2][j] == -1 and state[i + 3][j] == -1:
                    return -1

    for i in range(height):
        for j in range(width - 3):
            if state[i][j] == 1 and state[i][j + 1] == 1 and state[i][j + 2] == 1 and state[i][j + 3] == 1:
                return 1
            if state[i][j] == -1 and state[i][j + 1] == -1 and state[i][j + 2] == -1 and state[i][j + 3] == -1:
                return -1

    for i in range(height - 3):
        for j in range(width - 4, width):
            if state[i][j] == 1 and state[i + 1][j - 1] == 1 and state[i + 2][j - 2] == 1 and state[i + 3][j - 3] == 1:
                return 1
            if state[i][j] == -1 and state[i + 1][j - 1] == -1 and state[i + 2][j - 2] == -1 and state[i + 3][j - 3] == -1:
                return -1

    for i in range(height - 3):
        for j in range(width - 3):
            if state[i][j] == 1 and state[i + 1][j + 1] == 1 and state[i + 2][j + 2] == 1 and state[i + 3][j + 3] == 1:
                return 1
            if state[i][j] == -1 and state[i + 1][j + 1] == -1 and state[i + 2][j + 2] == -1 and state[i + 3][j + 3] == -1:
                return -1

    if sum(state[0] == 0) == 0:
        return 1e-4

    return 0
