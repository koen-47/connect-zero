from game.Game import Game
from brain.AlphaZero import AlphaZero
from experiment.Experiment import Experiment


game = Game()
alpha_zero = AlphaZero(game, n_iterations=128, n_episodes=100, n_games=20)
alpha_zero.start()
