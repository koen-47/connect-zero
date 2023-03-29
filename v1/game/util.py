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
    return board_state.copy()


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
        for j in (range(width - 4, width)):
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



def win_rate_test(model, device):
    game = Game()
    model = model.to(device)

    win_moves_taken_list = []
    win = []
    opponent_strat = AlphaBetaPruningStrategy(player_id=2, depth=3)

    for i in range(100):
        game.reset()
        win_moves_taken = 0

        while not game.is_game_over():
            state_1 = game.board.board
            action = opponent_strat.calculate_move(state_1)
            game.board.drop(1, action)

            if game.board.check_win() == 1:
                break

            action = select_action(game.board, model, device=device)
            game.board.drop(2, action)
            win_moves_taken += 1

            if game.board.check_win() == 2:
                win_moves_taken_list.append(win_moves_taken)
                win.append(1)
                break

    game.reset()
    num_moves_taken = len(win_moves_taken_list) if len(win_moves_taken_list) > 0 else 1
    return sum(win) / 100, sum(win_moves_taken_list) / num_moves_taken