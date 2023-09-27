from game.Game import Game
from brain.SelfPlay import SelfPlay
from brain.AlphaZero import AlphaZero


game = Game()
alpha_zero = AlphaZero(game, n_iterations=5, n_episodes=10, n_games=40)
alpha_zero.start()
