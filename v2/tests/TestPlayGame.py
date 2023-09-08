import unittest

import torch

from v2.game.Game import Game
from v2.game.Player import Player
from v2.game.Arena import Arena
from v2.strategy.ManualStrategy import ManualStrategy
from v2.strategy.RandomStrategy import RandomStrategy
from v2.strategy.AlphaZeroStrategy import AlphaZeroStrategy, MCTS
from v2.models.pytorch.ResNet import ResNet


class TestPlayGame(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Game()

    def test_play_manual_p1(self):
        g = Game()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nnet = ResNet(num_channels=128, num_res_blocks=5).to(device)
        nnet.load_state_dict(torch.load("../models/saved/resnet_1.pth"))
        mcts = MCTS(game=g, nnet=nnet, device=device, cpuct=1.)

        player_1 = Player(1, strategy=ManualStrategy())
        player_2 = Player(-1, strategy=AlphaZeroStrategy(mcts=mcts))
        arena = Arena(player_1, player_2)
        arena.play()

    def test_play_manual_p2(self):
        g = Game()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nnet = ResNet(num_channels=128, num_res_blocks=5).to(device)
        nnet.load_state_dict(torch.load("../models/saved/resnet_1.pth"))
        mcts = MCTS(game=g, nnet=nnet, device=device, cpuct=1.)

        player_1 = Player(1, strategy=AlphaZeroStrategy(mcts=mcts))
        player_2 = Player(-1, strategy=ManualStrategy())
        arena = Arena(player_1, player_2)
        arena.play()


    def test_play_random(self):
        g = Game()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nnet = ResNet(num_channels=128, num_res_blocks=5).to(device)
        nnet.load_state_dict(torch.load("../models/saved/resnet_1.pth"))
        mcts = MCTS(game=g, nnet=nnet, device=device, cpuct=1.)

        player_1 = Player(1, strategy=RandomStrategy())
        player_2 = Player(-1, strategy=AlphaZeroStrategy(mcts=mcts))
        arena = Arena(player_1, player_2)
        arena.play()
