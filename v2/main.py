from game.Game import Game
from brain.AlphaZero import AlphaZero


game = Game()
alpha_zero = AlphaZero(game, n_iterations=128, n_episodes=10, n_games=10)
alpha_zero.start()
