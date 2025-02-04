import numpy as np
from tqdm import tqdm

from game.Game import Game


class Evaluator:
    """
    Class to handle two players playing against each other for a specified number of games.
    """
    def __init__(self, player_1, player_2, logger=None):
        """
        Constructor for the Evaluator.
        :param player_1: instance of Player object for player 1.
        :param player_2: instance of Player object for player 2.
        :param logger: logger to record all results per iteration.
        """
        self.player_1 = player_1
        self.player_2 = player_2
        self.logger = logger

    def play_game(self, display=False):
        """
        Play a single game of Connect Four between the two players.
        :param display: flag to indicate if the board will be printed every turn.
        :return: tuples containing the result from playing the game and a history of the game in terms of
        (board, current player, action, policy).
        """
        game = Game()
        board = game.get_initial_board()
        status = 0
        player = self.player_1
        states = []

        # Start turn while no end game condition has been reached.
        while status == 0:
            # Flip state to perspective of current player.
            state = game.get_canonical_form(board, player.id)

            # Calculate move for current player based on that player's strategy.
            action, policy = player.strategy.calculate_move(state, player.id)

            # Play move and record history.
            board, _ = game.get_next_state(board, player.id, action)
            states.append((board, player.id, action, policy))
            player = self.player_2 if player.id == self.player_1.id else self.player_1

            # Check if game has ended.
            status = game.get_game_ended(board, player.id)

            # Display the board (if set).
            if display:
                print(game.display(board))

        result = player.id * status

        # Log result
        if self.logger is not None:
            self.logger.log(f"(Evaluation) Result: {result}", to_iteration=True)
            self.logger.log(f"{game.display(board, color=False)}", to_iteration=True)
        return result, states

    def play_games(self, n_games, return_states=False):
        """
        Play a specified number of games between two players (each player plays on each half for an equal number
        of games).
        :param n_games: number of games to play in total (not per half).
        :param return_states: flag to denote if a history of states should be returned.
        :return: list containing the results from playing all games.
        """
        def play_half(results, all_states, half):
            """
            Play one half of the specified number of games.
            :param results: results recorded so far across all games.
            :param all_states: history of states recorded so far across all games.
            :param half: number to denote which half it is (1 or 2).
            :return: list of updated results with new result.
            """
            for _ in tqdm(range(int(n_games/2)), desc=f"Evaluation ({half})"):
                result, states = self.play_game()
                results[int(np.rint(result)) + 1] += 1
                all_states.append((half, states, result))
            return results

        all_states = []
        results = [0, 0, 0]

        # Play first half
        results = play_half(results, all_states, 1)

        # Switch player 1 and 2
        self.player_1, self.player_2 = self.player_2, self.player_1

        # Play second half
        results = play_half(results, all_states, 2)

        if return_states:
            return results, all_states
        return results
