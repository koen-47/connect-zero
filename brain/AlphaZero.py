"""
File to handle the overall AlphaZero algorithm (self-play + evaluation loop)
"""

import copy
import json

import numpy as np
import torch

from brain.MCTS import MCTS
from logs.Logger import Logger
from models.DualResidualNetwork import DualResidualNetwork
from brain.Evaluator import Evaluator
from brain.SelfPlay import SelfPlay
from game.Player import Player
from strategy.AlphaZeroStrategy import AlphaZeroStrategy
from strategy.RandomStrategy import RandomStrategy
from strategy.AlphaBetaPruningStrategy import AlphaBetaPruningStrategy


class AlphaZero:
    def __init__(self, game, n_iterations=100, n_episodes=100, n_games=40):
        self.game = game
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.n_games = n_games
        self.logger = Logger(iteration_path="./logs/recent/log_iteration_1",
                             summary_path="./logs/recent/log_summary")

    def start(self):
        version = 4
        num_channels, num_res_blocks = 128, 5
        model_1 = DualResidualNetwork(num_channels=num_channels, num_res_blocks=num_res_blocks)
        model_2 = copy.deepcopy(model_1)
        training_examples = []

        for i in range(self.n_iterations):
            self.logger.set_log_iteration_file(num=i + 1, file=f"./logs/recent/log_iteration_{i + 1}")
            self.logger.log(f"Iteration {i + 1}", to_summary=True, to_iteration=True)

            self_play = SelfPlay(self.game, logger=self.logger)
            examples, results = self_play.play_episodes(model_1, n_episodes=self.n_episodes)
            training_examples.extend(examples)

            with open(f"./data/dataset_v{version}_{num_channels}_{num_res_blocks}.json", "w") as file:
                json.dump(list(training_examples), file)

            board_reward_states = np.array(training_examples, dtype=object)
            board_reward_states = list(zip(board_reward_states[:, 0], board_reward_states[:, 2]))
            board_reward_states = {str(board): reward for board, reward in board_reward_states}

            self.logger.log(f"(Self-play) Number of new training examples: {len(examples)}", to_summary=True,
                            to_iteration=True)
            self.logger.log(f"(Self-play) Number of total training examples: {len(training_examples)}",
                            to_summary=True, to_iteration=True)
            self.logger.log(f"(Self-play) Number of total unique state-reward pairs: {len(board_reward_states)}",
                            to_summary=True, to_iteration=True)
            self.logger.log(f"(Self-play) Model 1 wins: {results[2]}. Draws: {results[1]}. Model 2 wins: {results[0]}",
                            to_summary=True)

            model_2.train()
            model_2 = model_2.train_on_examples(training_examples, num_epochs=10, lr=0.0001, logger=self.logger)

            model_1.eval()
            model_2.eval()
            mcts_1 = MCTS(self.game, model_1, self.device, c_puct=2., dir_e=0.)
            mcts_2 = MCTS(self.game, model_2, self.device, c_puct=2., dir_e=0.)

            player_1 = Player(1, strategy=AlphaZeroStrategy(mcts=mcts_1))
            player_2 = Player(-1, strategy=AlphaZeroStrategy(mcts=mcts_2))
            evaluator = Evaluator(player_1, player_2, logger=self.logger)
            results = evaluator.play_games(self.n_games)
            n_model_2_wins, n_draws, n_model_1_wins = results
            model_2_win_rate = n_model_2_wins / sum(results)

            self.logger.log(f"(Evaluation) Win rate: {model_2_win_rate}", to_summary=True, to_iteration=True)
            self.logger.log(f"(Evaluation) Model 1 wins: {n_model_1_wins}. Draws: {n_draws}. "
                            f"Model 2 wins: {n_model_2_wins}", to_summary=True, to_iteration=True)

            if model_2_win_rate > 0.55:
                model_1 = copy.deepcopy(model_2)
                best_mcts = mcts_2
                print(f"Accepting new model...")
                self.logger.log("(Evaluation) Accepting new model...", to_summary=True, to_iteration=True)
            else:
                model_2 = copy.deepcopy(model_1)
                best_mcts = mcts_1
                print(f"Rejecting new model...")
                self.logger.log("(Evaluation) Rejecting new model...", to_summary=True, to_iteration=True)
            torch.save(model_2.state_dict(), f"./models/recent/resnet_v{version}_{num_channels}_{num_res_blocks}.pth")

            player_1 = Player(1, strategy=AlphaZeroStrategy(mcts=best_mcts))
            player_2 = Player(-1, strategy=RandomStrategy())
            evaluator = Evaluator(player_1, player_2, logger=self.logger)
            results = evaluator.play_games(self.n_games)
            n_random_strategy_wins, n_draws, n_alpha_zero_wins = results
            win_rate_against_random = n_alpha_zero_wins / sum(results)
            self.logger.log(f"(Experiment) Win rate (random): {win_rate_against_random}", to_summary=True,
                            to_iteration=True)
            print("Win rate (random):", win_rate_against_random)

            depth = 5
            player_1 = Player(1, strategy=AlphaZeroStrategy(mcts=best_mcts))
            player_2 = Player(-1, strategy=AlphaBetaPruningStrategy(depth=depth))
            evaluator = Evaluator(player_1, player_2, logger=self.logger)
            results = evaluator.play_games(self.n_games)
            n_alpha_beta_wins, n_draws, n_alpha_zero_wins = results
            win_rate_against_alpha_beta = n_alpha_zero_wins / sum(results)
            self.logger.log(f"(Experiment) Win rate (alpha-beta pruning with depth {depth}): "
                            f"{win_rate_against_alpha_beta}", to_summary=True, to_iteration=True)
            print(f"Win rate (alpha-beta pruning with depth {depth}):", win_rate_against_alpha_beta)
            self.logger.log("\n", to_summary=True)
