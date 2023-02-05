from game.board import Board
from game.game import Game


class MCTS:
    """
    Code for this class was adapted from https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
    """

    def __init__(self, game: Game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def search(self, board: Board):
        pass
