from game.Game import Game
from brain.AlphaZero import AlphaZero
from experiment.Experiment import Experiment


game = Game()
alpha_zero = AlphaZero(game, n_iterations=128, n_episodes=5, n_games=2)
alpha_zero.start()

