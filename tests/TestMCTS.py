import unittest

from game.game import Game
from strategies.mcts import MCTS


class TestMCTS(unittest.TestCase):
    def test_mcts_1(self):
        game = Game()
        mcts = MCTS(game=None, model=None, player_id=1, num_sims=25)
        mcts.get_action_probability(game.board.board, player_id=1, temp=0)
        game.board.drop(1, 1)
        mcts.get_action_probability(game.board.board, player_id=1, temp=0)
        game.board.drop(1, 2)
        mcts.get_action_probability(game.board.board, player_id=1, temp=0   )

