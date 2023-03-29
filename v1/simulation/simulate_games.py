import copy
from typing import List, Tuple
import json

import numpy as np
import pandas as pd

from v1.game.game import Game
from v1.strategies.strategy import AlphaBetaPruningStrategy


def simulate_games(num_games: int):
    game = Game()

    wins = [0, 0, 0]

    # (game_num, turn_num, player_turn, board_state, optimal_move, result)
    global_training_data = []
    for i in range(num_games):
        game.reset()

        if i % 100 == 0:
            print(i)

        turn_num = 1
        p1_strategy = AlphaBetaPruningStrategy(player_id=2, depth=3)
        p2_strategy = AlphaBetaPruningStrategy(player_id=1, depth=3)
        local_training_data = []
        while not game.is_game_over():
            p1_move = p1_strategy.calculate_move(game.board.board)
            p1_move_enc = np.zeros(7, dtype=int)
            p1_move_enc[p1_move] = 1
            p1_state = copy.deepcopy(game.board.board)
            game.board.drop(1, p1_move)
            local_training_data.append((p1_state, p1_move_enc.tolist()))

            game_status = game.board.check_win()
            if game_status != -1:
                break

            p2_move = p2_strategy.calculate_move(game.board.board)
            p2_move_enc = np.zeros(7)
            p2_move_enc[p2_move] = 1
            p2_state = copy.deepcopy(game.board.board)
            game.board.drop(2, p2_move)
            local_training_data.append((p2_state, p2_move_enc.tolist()))
            turn_num += 1

        game_status = game.board.check_win()
        print(np.array(game.board.board))
        print(game_status)
        if game_status == 2:
            wins[0] += 1
        elif game_status == 0:
            wins[1] += 1
        elif game_status == 1:
            wins[2] += 1
        # print(game_status)
        for data in local_training_data:
            data = data + (-1 if game_status == 2 else game_status,)
            global_training_data.append(data)

    # print(global_training_data)
    print(wins)
    # print(len(global_training_data))
    return global_training_data


def save_training_date_to_file(training_data: List[Tuple], file_path: str):
    df = pd.DataFrame.from_records(training_data, columns=["board", "policy", "value"])
    df.to_csv(file_path, encoding="utf-8")


def read_training_data(file_path: str):
    df = pd.read_csv(file_path)
    print(json.loads(df["board"].iloc[0]))
    print(df["policy"])
    print(df["value"])


training_data = simulate_games(100)
save_training_date_to_file(training_data, "../data/classification/raw_game_data_v2.csv")
# read_training_data("../data/classification/raw_game_data_v2.csv")
