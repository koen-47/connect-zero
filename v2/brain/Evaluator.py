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

        while status == 0:
            state = game.get_canonical_form(board, player.id)
            action = player.strategy.calculate_move(state, player.id)
            board, _ = game.get_next_state(board, player.id, action)
            player = self.player_2 if player.id == self.player_1.id else self.player_1
            status = game.get_game_ended(board, player.id)
            if display:
                print(game.display(board))

        result = player.id * status
        self.logger.log(f"(Evaluation) Result: {result}", to_iteration=True)
        self.logger.log(f"{game.display(board, color=False)}", to_iteration=True)
        return result

    def play_games(self, n_games):
        def play_half(results):
            for _ in tqdm(range(int(n_games/2)), desc="Evaluation"):
                result = self.play_game()
                results[int(np.rint(result)) + 1] += 1
            return results

        results = [0, 0, 0]
        results = play_half(results)
        print(results)
        self.player_1, self.player_2 = self.player_2, self.player_1
        results = play_half(results)
        return results
