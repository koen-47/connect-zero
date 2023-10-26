import copy

import torch

from v2.brain.MCTS import MCTS
from v2.logs.Logger import Logger
from v2.models.pytorch.DualResidualNetwork import DualResidualNetwork
from v2.brain.Evaluator import Evaluator
from v2.brain.SelfPlay import SelfPlay
from v2.game.Player import Player
from v2.strategy.AlphaZeroStrategyV2 import AlphaZeroStrategyV2 as AlphaZeroStrategy


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
        model_1 = DualResidualNetwork(num_channels=512, num_res_blocks=5)
        model_2 = DualResidualNetwork(num_channels=512, num_res_blocks=5)
        training_examples = []

        for i in range(self.n_iterations):
            self.logger.set_log_iteration_file(num=i + 1, file=f"./logs/recent/log_iteration_{i + 1}")
            self.logger.log(f"Iteration {i+1}", to_summary=True, to_iteration=True)
            self_play = SelfPlay(self.game, logger=self.logger)
            dataset, results = self_play.play_episodes(model_1, n_episodes=self.n_episodes)
            training_examples.extend(dataset.data)
            self.logger.log(f"(Self-play) Number of new training examples: {len(dataset.data)}", to_summary=True,
                            to_iteration=True)
            self.logger.log(f"(Self-play) Number of total training examples: {len(training_examples)}", to_summary=True,
                            to_iteration=True)
            self.logger.log(f"(Self-play) Model 1 wins: {results[2]}. Draws: {results[1]}. Model 2 wins: {results[0]}",
                            to_summary=True)

            model_2 = model_2.train_on_examples(training_examples, lr=0.0001, logger=self.logger)
            mcts_1 = MCTS(self.game, model_1, self.device)
            mcts_2 = MCTS(self.game, model_2, self.device)

            player_1 = Player(1, strategy=AlphaZeroStrategy(mcts=mcts_1))
            player_2 = Player(-1, strategy=AlphaZeroStrategy(mcts=mcts_2))
            evaluator = Evaluator(player_1, player_2, logger=self.logger)
            results = evaluator.play_games(self.n_games)
            n_model_2_wins, n_draws, n_model_1_wins = results
            model_2_win_rate = n_model_2_wins / sum(results)
            print(results)

            self.logger.log(f"(Evaluation) Win rate: {model_2_win_rate}", to_summary=True, to_iteration=True)
            self.logger.log(f"(Evaluation) Model 1 wins: {n_model_1_wins}. Draws: {n_draws}. "
                            f"Model 2 wins: {n_model_2_wins}", to_summary=True, to_iteration=True)

            if model_2_win_rate >= 0.55:
                model_1 = copy.deepcopy(model_2)
                print(f"Accepting new model...")
                self.logger.log("(Evaluation) Accepting new model...", to_summary=True, to_iteration=True)
            else:
                model_2 = copy.deepcopy(model_1)
                print(f"Rejecting new model...")
                self.logger.log("(Evaluation) Rejecting new model...", to_summary=True, to_iteration=True)
            torch.save(model_2.state_dict(), "./models/recent/resnet_small_v3.pth")
            self.logger.log("\n", to_summary=True)
