from game.Game import Game
from brain.SelfPlay import SelfPlay
from brain.AlphaZero import AlphaZero


game = Game()
alpha_zero = AlphaZero(game, n_iterations=80, n_episodes=4, n_games=4)
alpha_zero.start()
