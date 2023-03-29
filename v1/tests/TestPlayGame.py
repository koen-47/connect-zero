import unittest

from v1.game.game import Game


class TestPlayGame(unittest.TestCase):
    def setUp(self) -> None:
        self.game = Game()

    def test_play_manual(self):
        self.game.play_manual()
