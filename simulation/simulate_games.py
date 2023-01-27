import copy
from typing import List, Tuple
import json

import numpy as np
import pandas as pd

from game.game import Game
from game.player import Player
from strategies.strategy import AlphaBetaPruningStrategy


def simulate_games(num_games: int):
    game = Game()

    wins = [0, 0, 0]

    # (game_num, turn_num, player_turn, board_state, optimal_move, result)
    global_training_data = []
    for i in range(num_games):
        game.reset()

        turn_num = 1
        p1_strategy = AlphaBetaPruningStrategy(player_id=1, depth=3)
        p2_strategy = AlphaBetaPruningStrategy(player_id=2, depth=3)
        local_training_data = []
        while not game.is_game_over():
            p1_move = p1_strategy.calculate_move(game.board.board)
            p1_state = copy.deepcopy(game.board.board)
            game.board.drop(1, p1_move)
            local_training_data.append((i, turn_num, 1, p1_state, p1_move))

            game_status = game.board.check_win()
            if game_status != -1:
                break

            p2_move = p2_strategy.calculate_move(game.board.board)
            p2_state = copy.deepcopy(game.board.board)
            game.board.drop(2, p2_move)
            local_training_data.append((i, turn_num, 2, p2_state, p2_move))
            turn_num += 1

        game_status = game.board.check_win()
        if game_status == 2:
            wins[2] += 1
        elif game_status == 0:
            wins[1] += 1
        elif game_status == 1:
            wins[0] += 1
        # print(game_status)
        for data in local_training_data:
            data = data + (-1 if game_status == 2 else game_status,)
            global_training_data.append(data)

    # print(global_training_data)
    print(wins)
    # print(len(global_training_data))
    return global_training_data


def save_training_date_to_file(training_data: List[Tuple]):
    df = pd.DataFrame.from_records(training_data, columns=["game_num", "turn_num", "player_turn", "board_state",
                                                           "optimal_move", "result"])
    df.to_csv("../data/classification/game_data.csv", encoding="utf-8")


def read_training_data(file_path: str):
    df = pd.read_csv(file_path)
    print(json.loads(df["board_state"].iloc[0]))


# training_date = simulate_games(100)
# save_training_date_to_file(training_date)
