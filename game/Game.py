from .Board import Board


class Game:
    def __init__(self, board=None):
        self.board = board if board is not None else Board()

    def get_initial_board(self):
        return self.board.state

    def get_board_size(self):
        return self.board.height, self.board.width

    def get_action_size(self):
        return self.board.width

    def get_next_state(self, board, player_id, action):
        b = self.board.clone(state=board)
        b.move(player_id, action)
        return b.state, -player_id

    def get_valid_moves(self, board):
        return self.board.clone(board).get_valid_moves()

    def get_game_ended(self, board, player):
        board = self.board.clone(state=board)
        status = board.get_status()
        if status == player:
            return 1
        elif status == -player:
            return -1
        elif status == 1e-4:
            return 1e-4
        return 0

    def get_canonical_form(self, board, player):
        return board * player

    def display(self, board):
        board_str = "\n"
        for i in range(len(board)):
            for j in range(len(board[0])):
                cell_str = "- "
                if board[i][j] == 1:
                    cell_str = "X "
                elif board[i][j] == -1:
                    cell_str = "O "
                board_str += cell_str
            board_str += "\n"
        return board_str

