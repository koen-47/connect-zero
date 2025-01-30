import numpy as np
from abc import ABC

from strategy.Strategy import Strategy
from game.Game import Game


class AlphaBetaPruningStrategy(ABC, Strategy):
    def __init__(self, depth=4):
        super().__init__()
        self.depth = depth
        self.game = Game()

    def calculate_move(self, board, player_id):
        # FIX ISSUE WHERE PLAYER 1 AND 2 STRATEGY ARE NOT PERFORMING THE SAME
        return self.start_minimax(board, depth=self.depth, player=1)

    def start_minimax(self, board, depth, player):
        valid_moves = self.game.get_valid_moves(board)
        valid_moves = np.where(np.array(valid_moves) == 1)[0]
        np.random.shuffle(valid_moves)

        alpha = float("-inf")
        beta = float("inf")

        if player == 1:
            opponent = -1
        else:
            opponent = 1

        policy = np.zeros(len(board[0]))
        for move in valid_moves:
            next_state, next_player = self.game.get_next_state(board, player, move)
            score = self.minimize_beta(next_state, depth - 1, alpha, beta, player, opponent)
            policy[move] = score
        max_value = max(policy)
        max_indices = [i for i, value in enumerate(policy) if value == max_value]
        best_move = np.random.choice(max_indices) if len(max_indices) > 1 else max_indices[0]
        return best_move, policy

    def minimize_beta(self, board, depth, a, b, player, opponent):
        valid_moves = self.game.get_valid_moves(board)
        valid_moves = np.where(np.array(valid_moves) == 1)[0]

        game_status = self.game.get_game_ended(board, player)
        if depth == 0 or len(valid_moves) == 0 or game_status != 0:
            return self.__constant_reward(player, game_status)

        valid_moves = self.game.get_valid_moves(board)
        valid_moves = np.where(np.array(valid_moves) == 1)[0]
        beta = b

        for move in valid_moves:
            score = float("inf")
            if a < beta:
                next_state, next_player = self.game.get_next_state(board, opponent, move)
                score = self.maximize_alpha(next_state, depth - 1, a, beta, player, opponent)

            if score < beta:
                beta = score
        return beta

    def maximize_alpha(self, board, depth, a, b, player, opponent):
        valid_moves = self.game.get_valid_moves(board)
        valid_moves = np.where(np.array(valid_moves) == 1)[0]
        game_status = self.game.get_game_ended(board, player)
        if depth == 0 or len(valid_moves) == 0 or game_status != 0:
            return self.__constant_reward(player, game_status)

        alpha = a

        for move in valid_moves:
            score = float("-inf")
            if alpha < b:
                next_state, next_player = self.game.get_next_state(board, player, move)
                score = self.minimize_beta(next_state, depth - 1, alpha, b, player, opponent)
            if score > alpha:
                alpha = score
        return alpha

    def __constant_reward(self, player_id: int, game_status: int):
        if game_status == 1 and player_id == 1:
            return 1
        elif game_status == 1 and player_id == -1:
            return -1
        elif game_status == -1 and player_id == 1:
            return -1
        elif game_status == -1 and player_id == -1:
            return 1
        return 0.
