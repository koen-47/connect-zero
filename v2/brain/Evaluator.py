import numpy as np
from tqdm import tqdm

from v2.game.Game import Game


class Evaluator:
    def __init__(self, player_1, player_2, logger=None):
        self.player_1 = player_1
        self.player_2 = player_2
        self.logger = logger

    def play_game(self, display=False):
        game = Game()
        board = game.get_init_board()
        status = 0
        player = self.player_1
        states = []

        while status == 0:
            state = game.get_canonical_form(board, player.id)
            action, policy = player.strategy.calculate_move(state, player.id)
            board, _ = game.get_next_state(board, player.id, action)
            states.append((board, player.id, action, policy))
            player = self.player_2 if player.id == self.player_1.id else self.player_1
            status = game.get_game_ended(board, player.id)
            if display:
                print(game.display(board))

        result = player.id * status
        if self.logger is not None:
            self.logger.log(f"(Evaluation) Result: {result}", to_iteration=True)
            self.logger.log(f"{game.display(board, color=False)}", to_iteration=True)
        return result, states

    def play_games(self, n_games, return_states=False):
        def play_half(results, all_states):
            for _ in tqdm(range(int(n_games/2)), desc="Evaluation"):
                result, states = self.play_game()
                results[int(np.rint(result)) + 1] += 1
                all_states.append((states, result))
            return results

        all_states = []
        results = [0, 0, 0]
        results = play_half(results, all_states)
        self.player_1, self.player_2 = self.player_2, self.player_1
        results = play_half(results, all_states)
        if return_states:
            return results, all_states
        return results
