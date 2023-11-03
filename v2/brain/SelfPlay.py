import numpy as np
import torch
from tqdm import tqdm

from v2.game.Game import Game
from v2.game.Player import Player
from v2.brain.Arena import Arena
from v2.game.Board import Board
from v2.strategy.AlphaZeroStrategyV2 import AlphaZeroStrategyV2 as AlphaZeroStrategy
# from models.keras import DQN1
# from models.keras.DQN1 import Connect4NNet
from v2.models.pytorch.DualResidualNetwork import DualResidualNetwork
# from v2.strategy.AlphaZeroStrategyV2 import MCTS
from v2.brain.MCTS import MCTS
from v2.logs.Logger import Logger
from v2.brain.Dataset import Dataset
from v2.brain.Evaluator import Evaluator


class SelfPlay:
    def __init__(self, game, logger):
        self.game = game
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.dataset = Dataset()

    def play_episodes(self, model, n_episodes, temp_threshold=15):
        dataset = Dataset()
        results = [0] * 3

        for i in tqdm(range(n_episodes), desc="Self-play"):
            reward, n_turn, player = 0, 0, 1
            board = self.game.get_init_board()
            mcts = MCTS(self.game, model, self.device)
            while reward == 0:
                self.logger.log(f"(Self-play) Episode {i + 1}. Turn {n_turn + 1}.", to_iteration=True)
                state = self.game.get_canonical_form(board, player)
                temp = int(n_turn < temp_threshold)
                action, probs = mcts.get_action_prob(state, temp=temp, device=self.device)
                dataset.add(state, probs, player, with_symmetry=True)
                board, player = self.game.get_next_state(board, player, action)
                self.logger.log(f"Action: {action}. Policy: {np.round(np.array(probs), decimals=5)}", to_iteration=True)
                self.logger.log(self.game.display(board, color=False), to_iteration=True)
                reward = self.game.get_game_ended(board, -player)
                # if reward != 0:
                #     print(self.game.display(board))
                #     print(reward, player, Board(state=board).get_status())
                n_turn += 1
            results[int(np.rint(reward)) + 1] += 1
            dataset.set_rewards(reward, player)
        dataset.data = np.delete(dataset.data, 2, 1)
        print(dataset.data)
        dataset.shuffle()
        print(results)
        return dataset, results
