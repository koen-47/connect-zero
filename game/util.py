import copy
from typing import List


def drop(board_state: List[List[int]], player: int, col: int):
    height = len(board_state)
    i: int = height
    for j in range(height):
        if board_state[j][col] != 0:
            i = j
            break

    board_state[i - 1][col] = player
    return copy.deepcopy(board_state)


def get_valid_moves(board_state: List[List[int]]):
    valid_moves = []
    for i in range(len(board_state[0])):
        if board_state[0][i] == 0:
            valid_moves.append(i)
    return valid_moves


def check_win(board_state: List[List[int]]):
    height = len(board_state)
    width = len(board_state[0])

    for i in range(height):
        for j in range(width):
            if (i + 3) < height:
                if board_state[i][j] == 1 and board_state[i + 1][j] == 1 and board_state[i + 2][j] == \
                        1 and board_state[i + 3][j] == 1:
                    return 1
                elif board_state[i][j] == 2 and board_state[i + 1][j] == 2 and board_state[i + 2][j] == \
                        2 and board_state[i + 3][j] == 2:
                    return 2

    for i in range(height):
        for j in range(width - 3):
            if board_state[i][j] == 1 and board_state[i][j + 1] == 1 and board_state[i][j + 2] == \
                    1 and board_state[i][j + 3] == 1:
                return 1
            if board_state[i][j] == 2 and board_state[i][j + 1] == 2 and board_state[i][j + 2] == \
                    2 and board_state[i][j + 3] == 2:
                return 2

    for i in range(height - 3):
        for j in (range(width - 3, width)):
            if board_state[i][j] == 1 and board_state[i + 1][j - 1] == 1 and board_state[i + 2] \
                    [j - 2] == 1 and board_state[i + 3][j - 3] == 1:
                # print("1 diagonal right-left")
                return 1
            if board_state[i][j] == 2 and board_state[i + 1][j - 1] == 2 and board_state[i + 2] \
                    [j - 2] == 2 and board_state[i + 3][j - 3] == 2:
                # print("2 diagonal right-left")
                return 2

    for i in range(height - 3):
        for j in range(width - 3):
            if board_state[i][j] == 1 and board_state[i + 1][j + 1] == 1 and \
                    board_state[i + 2][j + 2] == 1 and board_state[i + 3][j + 3] == 1:
                # print("1 diagonal left-right")
                return 1
            if board_state[i][j] == 2 and board_state[i + 1][j + 1] == 2 and \
                    board_state[i + 2][j + 2] == 2 and board_state[i + 3][j + 3] == 2:
                # print("2 diagonal left-right")
                return 2

    if len(get_valid_moves(board_state)) == 0:
        return 0
    return -1


def is_valid_move(board_state: List[List[int]], col: int):
    return board_state[0][col] == 0
