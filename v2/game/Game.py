import numpy
from termcolor import colored

from v2.strategy.MCTSStrategy.MCTSStrategy import MCTSStrategy
from .Board import Board


class Game:
    def __init__(self, board=None):
        self.board = board if board is not None else Board()

    def get_init_board(self):
        return self.board.state

    def get_board_size(self):
        return self.board.height, self.board.width

    def get_action_size(self):
        return self.board.width

    def get_next_state(self, board, player_id, action):
        b = self.board.clone(state=board)
        b.move(player_id, action)
        # print(b.state)
        return b.state, -player_id

    def get_valid_moves(self, board):
        return self.board.clone(board).get_valid_moves()

    def get_game_ended(self, board, player):
        b = self.board.clone(state=board)
        status = b.get_status()
        if status is None:
            return 1e-4
        elif status == player:
            return 1
        elif status == -player:
            return -1
        return 0

    def get_canonical_form(self, board, player):
        return board * player

    def get_symmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def get_string_representation(self, board):
        return numpy.array2string(board)

    def display(self, board, color=True):
        board_str = "\n"
        for i in range(len(board)):
            for j in range(len(board[0])):
                cell_str = "- "
                if board[i][j] == 1:
                    cell_str = colored("X ", "red") if color is True else "X "
                elif board[i][j] == -1:
                    cell_str = colored("O ", "yellow") if color is True else "O "
                board_str += cell_str
            board_str += "\n"
        return board_str

# game = Game()
# game.play()

