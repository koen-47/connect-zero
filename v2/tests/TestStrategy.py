import unittest

import torch

from v2.brain.Evaluator import Evaluator
from v2.brain.MCTS import MCTS
from v2.game.Game import Game
from v2.game.Player import Player
from v2.models.pytorch.DualResidualNetwork import DualResidualNetwork
from v2.strategy.AlphaBetaPruningStrategy import AlphaBetaPruningStrategy
from v2.strategy.ManualStrategy import ManualStrategy
from v2.strategy.AlphaZeroStrategyV2 import AlphaZeroStrategyV2


class TestStrategy(unittest.TestCase):
    def test_alpha_beta_pruning_strategy_p1(self):
        player_1 = Player(1, strategy=AlphaBetaPruningStrategy(depth=4))
        player_2 = Player(-1, strategy=ManualStrategy())
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_alpha_beta_pruning_strategy_p2(self):
        player_1 = Player(1, strategy=ManualStrategy())
        player_2 = Player(-1, strategy=AlphaBetaPruningStrategy(depth=4))
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_alphazero_alphabeta_games(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DualResidualNetwork(num_channels=64, num_res_blocks=5).to(device)
        model.load_state_dict(torch.load("../models/saved/resnet_v1.pth"))

        mcts = MCTS(game=Game(), model=model, device=device, c_puct=1.)
        player_1 = Player(1, strategy=AlphaZeroStrategyV2(mcts=mcts))
        player_2 = Player(-1, strategy=AlphaBetaPruningStrategy(depth=1))
        evaluator = Evaluator(player_1, player_2)
        print(evaluator.play_games(40))
