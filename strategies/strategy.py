import copy
import random
from typing import List

import torch
from abc import ABC, abstractmethod
import numpy as np

from game.board import Board
import brain.rl.rewards as reward_funcs
from models.cnn.cnn_dqn_2 import CNN_DQN_2
from models.cnn.cnn_dqn_1 import CNN_DQN_1
from brain.classification.train_classifier import Classifier1
from brain.classification.train_test_data import split_board_state
from game.util import drop, check_win, get_valid_moves, is_valid_move


class Strategy:
    def __init__(self):
        pass

    @abstractmethod
    def calculate_move(self, board: List[List[int]]):
        pass

    def get_valid_moves(self, board: List[List[int]]):
        return [i for i, pos in enumerate(board[0]) if pos == 0]


class RandomStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, board: List[List[int]]):
        moves = self.get_valid_moves(board)
        return random.choice(moves)


class ManualStrategy(Strategy, ABC):
    def __init__(self):
        super().__init__()

    def calculate_move(self, board: List[List[int]]):
        return int(input("Enter a column to drop: "))-1


class RLStrategy(Strategy, ABC):
    def __init__(self):
        self.model = Classifier1()
        self.model.load_state_dict(torch.load("../models/saved/dqn_cnn_v2_1.pth"), strict=False)
        self.model.eval()

        super().__init__()

    def calculate_move(self, board: List[List[int]]):
        available_actions = self.get_valid_moves(board)
        state = torch.tensor(board, dtype=torch.float, device="cpu").unsqueeze(dim=0).unsqueeze(dim=0)

        with torch.no_grad():
            r_actions = self.model(state)[0].to("cpu").squeeze()
            state_action_values = [r_actions[action] for action in available_actions]
            argmax_action = np.argmax(state_action_values)
            greedy_action = available_actions[argmax_action]
            return greedy_action


class ClassificationStrategy(Strategy, ABC):
    def __init__(self):
        super(ClassificationStrategy, self).__init__()
        self.model = Classifier1()
        self.model.load_state_dict(torch.load("../brain/classification/connect_4.pth"), strict=False)
        self.model.eval()

    def calculate_move(self, board: List[List[int]]):
        available_actions = self.get_valid_moves(board)
        board = np.array(board).flatten()
        channel_p1 = np.where(board == 2, 0, board).reshape((6, 7))
        channel_p2 = np.where(board == 1, 0, board).reshape((6, 7))
        state = torch.tensor(np.array([channel_p1, channel_p2], dtype=np.float32)).unsqueeze(dim=0)

        with torch.no_grad():
            r_actions = self.model(state)[0, :]
            state_action_values = [r_actions[action] for action in available_actions]
            argmax_action = np.argmax(state_action_values)
            greedy_action = available_actions[argmax_action]
            return greedy_action


class AlphaBetaPruningStrategy(Strategy, ABC):
    def __init__(self, player_id, depth):
        super().__init__()
        self.player_id = player_id
        self.depth = depth

    def calculate_move(self, board: List[List[int]]):
        return self.start_minimax_2(board, self.depth, self.player_id)
        # return self.start_minimax_1(board)

    def start_minimax_1(self, board: List[List[int]]):
        valid_moves = get_valid_moves(board)
        random.shuffle(valid_moves)
        best_move = valid_moves[0]
        best_score = float("-inf")

        alpha = float("-inf")
        beta = float("inf")

        perspective = 2 if self.player_id == 1 else 1
        # print(self.player_id)
        for move in valid_moves:
            next_board = drop(copy.deepcopy(board), self.player_id, move)
            score = self.minimax(next_board, self.depth, alpha, beta, perspective)

            # print(f"move: {move}, score: {score}")
            # print(move)

            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def minimax(self, board: List[List[int]], depth: int, alpha: float, beta: float, player_id: int):
        valid_moves = get_valid_moves(board)
        # print(f"{player_id}, {depth}, {check_win(board)}, {reward_funcs.sequence_count_reward(board)}, {board}")
        if depth == 0 or check_win(board) != -1:
            # print(player_id)
            reward = reward_funcs.sequence_count_reward(board, player_id)
            # print(f"player_id: {player_id}, reward: {reward}, board: {board}")
            return reward

        # print(f"\ndepth {depth}")
        if player_id == 1:
            # random.shuffle(valid_moves)
            for move in valid_moves:
                score = float("-inf")
                if alpha < beta:
                    next_board = drop(copy.deepcopy(board), 1, move)
                    score = self.minimax(next_board, depth - 1, alpha, beta, 2)

                if score > alpha:
                    alpha = score

                # max_eval = max(max_eval, eval)
                # # print(f"p1: {move}, {eval}, {max_eval}, {next_board}")
                # alpha = max(alpha, eval)
                # if beta <= alpha:
                #     break
            return alpha
        else:
            # b = beta
            # random.shuffle(valid_moves)
            for move in valid_moves:
                score = float("inf")
                if alpha < beta:
                    next_board = drop(copy.deepcopy(board), 2, move)
                    score = self.minimax(next_board, depth - 1, alpha, beta, 1)

                if score < beta:
                    beta = score
                # min_eval = min(min_eval, eval)
                # # print(f"p2: {move}, {eval}, {min_eval}, {next_board}")
                # beta = min(beta, eval)
                # if beta <= alpha:
                #     break
            return beta

    def start_minimax_2(self, board, depth, player):
        # get array of possible moves
        validMoves = get_valid_moves(board)
        random.shuffle(validMoves)
        bestMove = validMoves[0]
        bestScore = float("-inf")

        # initial alpha & beta values for alpha-beta pruning
        alpha = float("-inf")
        beta = float("inf")

        # print(f"player: {player}")

        if player == 1:
            opponent = 2
        else:
            opponent = 1

        # print(f"opponent: {opponent}")

        # go through all of those boards
        for move in validMoves:
            # create new board from move
            tempBoard = drop(copy.deepcopy(board), player, move)
            # call min on that new board
            boardScore = self.minimizeBeta(tempBoard, depth - 1, alpha, beta, player, opponent)
            # print(f"move: {move}, score: {boardScore}")

            if boardScore > bestScore:
                bestScore = boardScore
                bestMove = move
        # print(f"best_move: {bestMove}, best_score: {bestScore}")
        return bestMove

    def minimizeBeta(self, board, depth, a, b, player, opponent):
        validMoves = []
        for col in range(7):
            # if column col is a legal move...
            if is_valid_move(board, col):
                # make the move in column col for curr_player
                temp = drop(copy.deepcopy(board), player, col)
                validMoves.append(col)

        # check to see if game over
        if depth == 0 or len(validMoves) == 0 or check_win(board) != -1:
            # print(player)
            return reward_funcs.sequence_count_reward(board, player)

        validMoves = get_valid_moves(board)
        beta = b

        # if end of tree evaluate scores
        for move in validMoves:
            boardScore = float("inf")
            # else continue down tree as long as ab conditions met
            if a < beta:
                tempBoard = drop(copy.deepcopy(board), opponent, move)
                boardScore = self.maximizeAlpha(tempBoard, depth - 1, a, beta, player, opponent)

            if boardScore < beta:
                beta = boardScore
        return beta

    def maximizeAlpha(self, board, depth, a, b, player, opponent):
        validMoves = []
        for col in range(7):
            # if column col is a legal move...
            if is_valid_move(board, col):
                # make the move in column col for curr_player
                temp = drop(copy.deepcopy(board), player, col)
                validMoves.append(col)

        if depth == 0 or len(validMoves) == 0 or check_win(board) != -1:
            # print(player)
            return reward_funcs.sequence_count_reward(board, player)

        alpha = a
        # if end of tree, evaluate scores
        for move in validMoves:
            boardScore = float("-inf")
            if alpha < b:
                tempBoard = drop(copy.deepcopy(board), player, move)
                boardScore = self.minimizeBeta(tempBoard, depth - 1, alpha, b, player, opponent)

            if boardScore > alpha:
                alpha = boardScore
        return alpha
