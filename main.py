"""
File to run main.py
"""

import argparse

from game.Game import Game
from brain.AlphaZero import AlphaZero
from experiment.Experiment import Experiment


def main():
    # Setting up arguments for main.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--model", type=str, default="./models/recent/resnet_v4_128_5.pth")
    parser.add_argument("--n_games", type=int, default=100)

    # Parsing arguments
    args = parser.parse_args()
    is_training = args.train
    is_experimenting = args.experiment
    model_experiment = args.model
    n_games_experiment = args.n_games

    # Handling error for when --train and --experiment are both set
    if is_training and is_experimenting:
        print("ERROR: unable to train new model and run experiments simultaneously...")
    # Train model with default parameters
    elif is_training:
        game = Game()
        alpha_zero = AlphaZero(game, n_iterations=200, n_episodes=250, n_games=50)
        alpha_zero.start()
    # Experiment with trained model (or default model resnet_v2_512_2.pth) for a specified number of games
    elif is_experimenting:
        experiment = Experiment(model_experiment)
        # experiment.run(n_games_experiment, log_losses=True)
        experiment.plot_result_curves()


if __name__ == "__main__":
    main()
