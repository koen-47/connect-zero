import unittest
import numpy as np

from v1.game.game import Game
from v1.mcts import MCTS, execute_episode_mcts


class TestMCTS(unittest.TestCase):
    def test_mcts_1(self):
        game = Game()
        mcts = MCTS(model=None, player_id=1, num_sims=25)
        # mcts.get_action_probability(game.board.board, player_id=1, temp=0)
        game.board.drop(1, 5)
        game.board.drop(2, 1)
        probs = mcts.get_action_probability(game.board.board, player_id=1, temp=0)
        move = np.argmax(probs)
        print(move)

    def test_mcts_episode_1(self):
        execute_episode_mcts(40)
