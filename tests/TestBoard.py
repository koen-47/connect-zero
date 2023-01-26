import unittest

import numpy

from game.board import Board

from board_presets.presets import board_presets


class TestBoardComplete(unittest.TestCase):
    def test_vertical_basic_player1_win_1(self):
        board = Board(5, 5)
        board.set_board(board_presets["vertical"]["player1_win_1"])
        self.assertEqual(board.check_win(), 1)

    def test_vertical_basic_player1_win_2(self):
        board = Board(5, 5)
        board.set_board(board_presets["vertical"]["player1_win_2"])
        self.assertEqual(board.check_win(), 1)

    def test_vertical_basic_player2_win_1(self):
        board = Board(5, 5)
        board.set_board(board_presets["vertical"]["player2_win_1"])
        self.assertEqual(board.check_win(), 2)

    def test_vertical_basic_player2_win_2(self):
        board = Board(5, 5)
        board.set_board(board_presets["vertical"]["player2_win_2"])
        self.assertEqual(board.check_win(), 2)

    def test_horizontal_basic_player1_win_1(self):
        board = Board(5, 5)
        board.set_board(board_presets["horizontal"]["player1_win_1"])
        self.assertEqual(board.check_win(), 1)

    def test_horizontal_basic_player1_win_2(self):
        board = Board(5, 5)
        board.set_board(board_presets["horizontal"]["player1_win_2"])
        self.assertEqual(board.check_win(), 1)

    def test_horizontal_basic_player2_win_1(self):
        board = Board(5, 5)
        board.set_board(board_presets["horizontal"]["player2_win_1"])
        self.assertEqual(board.check_win(), 2)

    def test_horizontal_basic_player2_win_2(self):
        board = Board(5, 5)
        board.set_board(board_presets["horizontal"]["player2_win_2"])
        self.assertEqual(board.check_win(), 2)

    def test_diagonal_left_right_basic_player1_win_1(self):
        board = Board(5, 5)
        board.set_board(board_presets["diagonal_left_right"]["player1_win_1"])
        self.assertEqual(board.check_win(), 1)

    def test_diagonal_left_right_basic_player1_win_2(self):
        board = Board(5, 5)
        board.set_board(board_presets["diagonal_left_right"]["player1_win_2"])
        self.assertEqual(board.check_win(), 1)

    def test_diagonal_left_right_basic_player2_win_1(self):
        board = Board(5, 5)
        board.set_board(board_presets["diagonal_left_right"]["player2_win_1"])
        self.assertEqual(board.check_win(), 2)

    def test_diagonal_left_right_basic_player2_win_2(self):
        board = Board(5, 5)
        board.set_board(board_presets["diagonal_left_right"]["player2_win_2"])
        self.assertEqual(board.check_win(), 2)

    def test_diagonal_right_left_basic_player1_win_1(self):
        board = Board(5, 5)
        board.set_board(board_presets["diagonal_right_left"]["player1_win_1"])
        self.assertEqual(board.check_win(), 1)

    def test_diagonal_right_left_basic_player1_win_2(self):
        board = Board(5, 5)
        board.set_board(board_presets["diagonal_right_left"]["player1_win_2"])
        self.assertEqual(board.check_win(), 1)

    def test_diagonal_right_left_basic_player2_win_1(self):
        board = Board(5, 5)
        board.set_board(board_presets["diagonal_right_left"]["player2_win_1"])
        self.assertEqual(board.check_win(), 2)

    def test_diagonal_right_left_basic_player2_win_2(self):
        board = Board(5, 5)
        board.set_board(board_presets["diagonal_right_left"]["player2_win_2"])
        self.assertEqual(board.check_win(), 2)

    def test_realistic_1(self):
        board = Board(6, 7, board=board_presets["realistic"][0])
        game_status = board.check_win()
        print(game_status)
        print(numpy.array(board.board))
        self.assertEqual(-1, game_status)

    def test_realistic_2(self):
        board = Board(6, 7, board=board_presets["realistic"][1])
        game_status = board.check_win()
        print(game_status)
        print(numpy.array(board.board))
        self.assertEqual(-1, game_status)

    def test_realistic_3(self):
        board = Board(6, 7, board=board_presets["realistic"][2])
        game_status = board.check_win()
        print(game_status)
        print(numpy.array(board.board))
        self.assertEqual(-1, game_status)

    def test_realistic_4(self):
        board = Board(6, 7, board=board_presets["realistic"][3])
        game_status = board.check_win()
        print(game_status)
        print(numpy.array(board.board))
        self.assertEqual(-1, game_status)


class TestBoardSequential(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board(5, 5)

    def test_vertical_basic_player1_win(self):
        self.board.drop(1, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 0)
        self.assertEqual(self.board.check_win(), 1)

    def test_vertical_basic_player2_win(self):
        self.board.drop(2, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 0)
        self.assertEqual(self.board.check_win(), 2)

    def test_vertical_random_player2_win(self):
        self.board.drop(2, 0)
        self.board.drop(1, 1)
        self.board.drop(2, 0)
        self.board.drop(2, 3)
        self.board.drop(1, 2)
        self.board.drop(2, 0)
        self.board.drop(1, 2)
        self.board.drop(2, 0)
        self.assertEqual(self.board.check_win(), 2)

    def test_horizontal_basic_player1_win(self):
        self.board.drop(1, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 1)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 2)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 3)
        self.assertEqual(self.board.check_win(), 1)

    def test_horizontal_basic_player2_win(self):
        self.board.drop(2, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 1)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 2)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 3)
        self.assertEqual(self.board.check_win(), 2)

    def test_diagonal_lf_basic_player1_win(self):
        self.board.drop(1, 0)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 1)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 1)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 2)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 2)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 2)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 3)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 3)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(2, 3)
        self.assertEqual(self.board.check_win(), 0)
        self.board.drop(1, 3)

        print(self.board.to_string())

        self.assertEqual(self.board.check_win(), 1)
