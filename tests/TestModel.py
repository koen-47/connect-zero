import unittest

import numpy as np
import torch

from brain.MCTS import MCTS
from game.Game import Game
from game.Player import Player
from models.DualResidualNetwork import DualResidualNetwork
from game.Board import encode_board


class TestModel(unittest.TestCase):
    def test_value_prediction_1(self):
        board = np.array([[ 0,  0,  0,  0,  0, 0, 0],
                          [ 0,  0,  0,  0,  0, 0, 0],
                          [ 0,  0,  0,  0,  0, 0, 0],
                          [ 0,  0,  0,  1,  0, 0, 0],
                          [ 0,  0,  0,  1,  0, 0, 0],
                          [ 0, -1, -1,  1,  1, 0, 0]])
        board = encode_board(board, -1)

        device = torch.device("cpu")
        model = DualResidualNetwork(num_channels=128, num_res_blocks=5).to(device)
        model.load_state_dict(torch.load("../models/recent/resnet_v4_128_5.pth"))
        model.eval()

        tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
        tensor_state = tensor_state.unsqueeze(dim=0).to(device)
        policy, value = model(tensor_state)
        print(policy, value)

    def test_value_prediction_2(self):
        board = np.array([[ 0,  0,  0,  0,  0,  0,  0],
                          [ 0,  0,  0,  0,  0,  0,  0],
                          [ 1,  0,  0,  0,  0,  0,  0],
                          [-1, -1, -1,  0,  0,  1,  0],
                          [ 1,  1, -1, -1,  0,  1,  0],
                          [ 1, -1,  1, -1, -1,  1,  0]])
        board = encode_board(board, -1)

        device = torch.device("cpu")
        model = DualResidualNetwork(num_channels=128, num_res_blocks=5).to(device)
        model.load_state_dict(torch.load("../models/recent/resnet_v4_128_5.pth"))
        model.eval()

        tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
        tensor_state = tensor_state.unsqueeze(dim=0).to(device)
        policy, value = model(tensor_state)
        print(policy, value)

    def test_value_prediction_3(self):
        board = np.array([[ 0,  0,  0,  0,  0,  0,  0],
                          [ 0,  1,  0, -1, -1, -1,  0],
                          [ 0,  1,  0, -1,  1,  1,  0],
                          [ 0, -1,  1,  1,  1, -1,  0],
                          [-1,  1,  1,  1, -1,  1, -1],
                          [ 1, -1, -1,  1, -1, -1, -1]])
        board = encode_board(board, 1)

        device = torch.device("cpu")
        model = DualResidualNetwork(num_channels=128, num_res_blocks=5).to(device)
        model.load_state_dict(torch.load("../models/recent/resnet_v4_128_5.pth"))
        model.eval()

        tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
        tensor_state = tensor_state.unsqueeze(dim=0).to(device)
        policy, value = model(tensor_state)
        print(policy, value)

    def test_play_game(self):
        self.play_game(display=True)

    def play_game(self, display=False):
        game = Game()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DualResidualNetwork(num_channels=512, num_res_blocks=2).to(device)
        model.load_state_dict(torch.load("../models/recent/resnet_128_2.pth"))
        mcts = MCTS(game=game, model=model, device=device, c_puct=1., dir_e=0)

        # player_1 = Player(1, strategy=RandomStrategy())
        # player_2 = Player(-1, strategy=AlphaZeroStrategyV2(mcts=mcts))

        player_1 = Player(1, strategy=AlphaZeroStrategyV2(mcts=mcts))
        player_2 = Player(-1, strategy=AlphaZeroStrategyV2(mcts=mcts))

        board = game.get_initial_board()
        status = 0
        player = player_1
        states = []

        while status == 0:
            state = game.get_canonical_form(board, player.id)
            action, policy = player.strategy.calculate_move(state, player.id)
            board, _ = game.get_next_state(board, player.id, action)
            states.append((board, player.id, action, policy))
            player = player_2 if player.id == player_1.id else player_1
            status = game.get_game_ended(board, player)
            if display:
                print(game.display(board))
                # print(board)
                tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
                tensor_state = tensor_state.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
                policy, value = model(tensor_state)
                print(policy, value)

        print(status)