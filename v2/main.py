from game.Game import Game
from brain.AlphaZero import AlphaZero


game = Game()
alpha_zero = AlphaZero(game, n_iterations=128, n_episodes=2, n_games=2)
alpha_zero.start()
