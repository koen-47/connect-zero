import unittest
import json
from typing import Dict

import brain.rl.rewards as reward_funcs


class TestRewardFunctions(unittest.TestCase):
    def test_count_sequences_horizontal_1(self):
        board = [[1, 1, 0, 0, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 2, 2, 2],
                 [0, 0, 0, 2, 2, 2, 2],
                 [1, 1, 1, 1, 0, 0, 0]]

        sequence_count = reward_funcs.sequence_count_reward(board)
        print_dict(sequence_count)

    def test_count_sequences_vertical_1(self):
        board = [[0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 1, 1],
                 [1, 0, 0, 0, 0, 1, 1],
                 [1, 0, 0, 0, 0, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0]]

        sequence_count = reward_funcs.sequence_count_reward(board)
        print_dict(sequence_count)

    def test_count_sequences_lr_diag_1(self):
        board = [[1, 0, 0, 1, 0, 0, 0],
                 [1, 1, 0, 0, 1, 0, 0],
                 [0, 1, 1, 1, 0, 1, 0],
                 [0, 0, 1, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]]

        sequence_count = reward_funcs.sequence_count_reward(board)
        print_dict(sequence_count)

    def test_count_sequences_rl_diag_1(self):
        board = [[0, 1, 0, 1, 0, 0, 1],
                 [1, 0, 1, 0, 0, 1, 0],
                 [0, 1, 0, 1, 1, 0, 1],
                 [1, 0, 1, 1, 0, 1, 0],
                 [0, 1, 0, 0, 1, 0, 1],
                 [1, 0, 0, 1, 0, 1, 0]]

        sequence_count = reward_funcs.sequence_count_reward(board)
        print_dict(sequence_count)


def print_dict(d: Dict):
    print(json.dumps(d, indent=3))


# board = [[0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0, 0, 0]]
