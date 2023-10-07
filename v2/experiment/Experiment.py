import numpy as np
import torch

from v2.game.Game import Game
from v2.game.Player import Player
from v2.brain.Arena import Arena
from v2.brain.Evaluator import Evaluator
from v2.strategy.AlphaBetaPruningStrategy import AlphaBetaPruningStrategy
from v2.strategy.ManualStrategy import ManualStrategy
from v2.strategy.RandomStrategy import RandomStrategy
from v2.strategy.AlphaZeroStrategyV2 import AlphaZeroStrategyV2
from v2.brain.MCTS import MCTS
from v2.models.pytorch.DualResidualNetwork import DualResidualNetwork
from v2.logs.Logger import Logger


class Experiment:
    def __init__(self, model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ = DualResidualNetwork(num_channels=128, num_res_blocks=8).to(device)
        model_.load_state_dict(torch.load(model))
        self.__mcts = MCTS(game=Game(), model=model_, device=device, num_sims=250, c_puct=1., dir_e=0)
        self.__logger = Logger()

    def run(self, n_games, log_losses=True):
        strategies = {"random": RandomStrategy()}
        for i in range(2, 6):
            strategies[f"alphabeta_{i}"] = AlphaBetaPruningStrategy(depth=i)

        for name, strategy_2 in strategies.items():
            player_1 = Player(1, strategy=AlphaZeroStrategyV2(mcts=self.__mcts))
            player_2 = Player(-1, strategy=strategy_2)
            evaluator = Evaluator(player_1, player_2)
            results, states = evaluator.play_games(n_games, return_states=True)
            n_player_2_wins, n_draws, n_player_1_wins = results
            win_rate = n_player_1_wins / sum(results)
            print(win_rate, results)

            if log_losses:
                self.__logger.set_log_experiment_file(name, f"../experiment/logs/experiment_{name}")
                self.__logger.log(f"Win rate: {win_rate}", to_experiment=True)
                self.__logger.log(f"Wins: {n_player_1_wins}. Draws: {n_draws}. Losses: {n_player_2_wins}.\n",
                                  to_experiment=True)

                self.__logger.log("Losses")
                losses = [state for state, result in states if result == -1]
                for i, loss in enumerate(losses):
                    for j, (state, player_id, action, policy) in enumerate(loss):
                        self.__logger.log(f"Loss: {i + 1}. Turn {j + 1} (player: {player_id})", to_experiment=True)
                        self.__logger.log(f"Action: {action}. Policy: {policy}", to_experiment=True)
                        self.__logger.log(f"{Game().display(state)}", to_experiment=True)
