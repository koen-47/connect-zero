import unittest

import numpy as np
import torch

from brain.Evaluator import Evaluator
from brain.MCTS import MCTS
from game.Game import Game
from game.Player import Player
from models.DualResidualNetwork import DualResidualNetwork
from strategy.AlphaBetaPruningStrategy import AlphaBetaPruningStrategy
from strategy.ManualStrategy import ManualStrategy
from strategy.AlphaZeroStrategy import AlphaZeroStrategy


class TestStrategy(unittest.TestCase):
    def test_alpha_beta_pruning_strategy_p1(self):
        player_1 = Player(1, strategy=AlphaBetaPruningStrategy(depth=1))
        player_2 = Player(-1, strategy=ManualStrategy())
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_alpha_beta_pruning_strategy_p2(self):
        player_1 = Player(1, strategy=ManualStrategy())
        player_2 = Player(-1, strategy=AlphaBetaPruningStrategy(depth=4))
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_alpha_beta_reward(self):
        strategy = AlphaBetaPruningStrategy(depth=1)
        board = np.array([[ 0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0],
                          [ 0,  1,  0, -1,  0,  0,  0],
                          [ 0,  1,  0,  1, -1,  0,  0],
                          [ 0,  1,  0,  1,  1, -1,  0],
                          [ 1,  1,  0,  1,  1, -1, -1]])
        strategy.reward(board, 1)

    def test_alphazero_alphabeta_games(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DualResidualNetwork(num_channels=64, num_res_blocks=5).to(device)
        model.load_state_dict(torch.load("../models/recent/resnet_v1.pth"))

        mcts = MCTS(game=Game(), model=model, device=device, c_puct=1.)
        player_1 = Player(1, strategy=AlphaZeroStrategyV2(mcts=mcts))
        player_2 = Player(-1, strategy=AlphaBetaPruningStrategy(depth=1))
        evaluator = Evaluator(player_1, player_2)
        print(evaluator.play_games(40))
