from abc import ABC

import numpy as np

from v2.strategy.Strategy import Strategy


class ManualStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, board, player_id):
        move = int(input("Enter a column to drop: "))-1
        policy = np.zeros(len(board[0]))
        policy[move] = 1
        # print(policy)
        return move, policy

