import unittest

import torch

from game.Game import Game
from game.Player import Player
from brain.Evaluator import Evaluator
from strategy.ManualStrategy import ManualStrategy
from strategy.RandomStrategy import RandomStrategy
from strategy.AlphaZeroStrategy import AlphaZeroStrategy
from strategy.AlphaBetaPruningStrategy import AlphaBetaPruningStrategy
from brain.MCTS import MCTS
from models.DualResidualNetwork import DualResidualNetwork


class TestPlayGame(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Game()

    def test_play_manual_p1(self):
        g = Game()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DualResidualNetwork(num_channels=128, num_res_blocks=8).to(device)
        model.load_state_dict(torch.load("../models/saved/resnet_128_8_71.pth"))
        mcts = MCTS(game=g, model=model, device=device, c_puct=1., dir_e=0)

        player_1 = Player(1, strategy=ManualStrategy())
        player_2 = Player(-1, strategy=AlphaZeroStrategyV2(mcts=mcts))
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_play_manual_p2(self):
        g = Game()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = DualResidualNetwork(num_channels=128, num_res_blocks=8).to(device)
        model = DualConvolutionalNetwork(num_channels=64).to(device)
        model.load_state_dict(torch.load("../models/recent/cnn_v1.pth"))
        mcts = MCTS(game=g, model=model, device=device, c_puct=1.)

        player_1 = Player(1, strategy=AlphaZeroStrategyV2(mcts=mcts))
        player_2 = Player(-1, strategy=ManualStrategy())
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_play_manual_random_p1(self):
        player_1 = Player(1, strategy=ManualStrategy())
        player_2 = Player(-1, strategy=AlphaBetaPruningStrategy(depth=5))
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_play_manual_random_p2(self):
        player_1 = Player(1, strategy=AlphaBetaPruningStrategy(depth=5))
        player_2 = Player(-1, strategy=ManualStrategy())
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_play_random_random_games(self):
        player_1 = Player(1, strategy=RandomStrategy())
        player_2 = Player(-1, strategy=RandomStrategy())
        evaluator = Evaluator(player_1, player_2)
        results, states = evaluator.play_games(4, return_states=True)

    def test_play_alphazero_random_1_game(self):
        g = Game()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DualResidualNetwork(num_channels=128, num_res_blocks=8).to(device)
        model.load_state_dict(torch.load("../models/recent/resnet_v4.pth"))
        mcts = MCTS(game=g, model=model, device=device, num_sims=250, c_puct=1., dir_e=0)

        player_1 = Player(1, strategy=AlphaZeroStrategyV2(mcts=mcts))
        player_2 = Player(-1, strategy=RandomStrategy())
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    def test_play_alphazero_random_500_games(self):
        g = Game()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DualResidualNetwork(num_channels=128, num_res_blocks=8).to(device)
        model.load_state_dict(torch.load("../models/recent/resnet_v4.pth"))
        mcts = MCTS(game=g, model=model, device=device, num_sims=250, c_puct=1., dir_e=0)

        player_1 = Player(1, strategy=AlphaZeroStrategyV2(mcts=mcts))
        player_2 = Player(-1, strategy=RandomStrategy())
        evaluator = Evaluator(player_1, player_2)
        results, states = evaluator.play_games(6, return_states=True)
        print(results)
        print([state for state, result in states if result == -1])
