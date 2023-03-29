from tqdm import tqdm

from .Game import Game
from .Player import Player
from v2.strategy.ManualStrategy import ManualStrategy
from v2.strategy.AlphaZeroStrategy import AlphaZeroStrategy
from v2.strategy.RandomStrategy import RandomStrategy


class Arena:
    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2

    def play(self):
        game = Game()
        board = game.get_init_board()
        players = [self.player_2, None, self.player_1]
        current_player = 1

        while game.get_game_ended(board, current_player) == 0:
            action = players[current_player + 1].strategy.calculate_move(game.get_canonical_form(board, current_player),
                                                                         current_player)
            board, current_player = game.get_next_state(board, current_player, action)

        #     print(game.get_valid_moves(board))
        #     print(board)
        #
        print(board)
        return current_player * game.get_game_ended(board, current_player)

    def play_games(self, num_games=100):
        num = int(num_games / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.play()
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player_1, self.player_2 = self.player_2, self.player_1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.play()
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws


# player1 = Player(1, strategy=ManualStrategy())
# player2 = Player(-1, strategy=RandomStrategy())
# arena = Arena(player_1=player1, player_2=player2)
# arena.play()
