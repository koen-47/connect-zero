from game.Game import Game
from brain.AlphaZero import AlphaZero


game = Game()
alpha_zero = AlphaZero(game, n_iterations=128, n_episodes=2500, n_games=40)
alpha_zero.start()
