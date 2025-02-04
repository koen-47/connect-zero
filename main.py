"""
File to run main.py
"""

import argparse

import torch

from brain.Evaluator import Evaluator
from brain.MCTS import MCTS
from game.Game import Game
from brain.AlphaZero import AlphaZero
from experiment.Experiment import Experiment
from game.Player import Player
from models.DualResidualNetwork import DualResidualNetwork
from strategy.AlphaZeroStrategy import AlphaZeroStrategy
from strategy.ManualStrategy import ManualStrategy


def main():
    # Setting up arguments for main.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", type=int)
    parser.add_argument("--experiment", type=int)

    # Parsing arguments
    args = parser.parse_args()
    player = args.play
    n_games = args.experiment

    is_playing = player is not None
    is_experimenting = n_games is not None

    # Handling error for when --play and --experiment are both set
    if is_playing and is_experimenting:
        print("ERROR: unable to play against model and run experiments simultaneously...")

    # Play against AlphaZero
    elif is_playing:
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DualResidualNetwork(num_channels=128, num_res_blocks=5).to(device)
        model.load_state_dict(torch.load("./models/saved/resnet_v4_128_5.pth", weights_only=True))
        mcts = MCTS(game=Game(), model=model, device=device, num_sims=100, c_puct=2.0, dir_e=0)

        # Set up strategies
        alphazero_strategy = AlphaZeroStrategy(mcts)
        player_strategy = ManualStrategy()

        if player == 1:
            player_1 = Player(1, strategy=player_strategy)
            player_2 = Player(-1, strategy=alphazero_strategy)
        else:
            player_1 = Player(1, strategy=alphazero_strategy)
            player_2 = Player(-1, strategy=player_strategy)

        # Play against AlphaZero
        evaluator = Evaluator(player_1, player_2)
        evaluator.play_game(display=True)

    # Run experiments on AlphaZero (saves logs under experiment/logs/recent)
    elif is_experimenting:
        experiment = Experiment("./models/saved/resnet_v4_128_5.pth")
        experiment.run(n_games, log_losses=True)


if __name__ == "__main__":
    main()
