import numpy as np
import torch
from tqdm import tqdm

from brain.MCTS import MCTS
from game.Board import encode_board


class SelfPlay:
    def __init__(self, game, logger):
        self.game = game
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger

    def play_episodes(self, model, n_episodes, temp_threshold=15):
        examples = []
        results = [0] * 3

        for i in tqdm(range(n_episodes), desc="Self-play"):
            reward, n_turn, player = 0, 0, 1
            board = self.game.get_initial_board()
            mcts = MCTS(self.game, model, self.device, c_puct=2., dir_alpha=0.5)
            episode_examples = []

            while reward == 0:
                self.logger.log(f"(Self-play) Episode {i + 1}. Turn {n_turn + 1}.", to_iteration=True)
                state = self.game.get_canonical_form(board, player)
                action, probs = mcts.get_action_prob(state, temp=int(n_turn < temp_threshold), device=self.device)

                encoded_state = encode_board(state, player)
                episode_examples += [[encoded_state, probs, player, None]]
                episode_examples += [[np.flip(encoded_state, 2).tolist(), np.flip(probs), player, None]]

                board, player = self.game.get_next_state(board, player, action)
                self.logger.log(f"Action: {action}. Policy: {np.round(np.array(probs), decimals=4)}", to_iteration=True)
                self.logger.log(self.game.display(board, color=False), to_iteration=True)
                reward = self.game.get_game_ended(board, player)
                n_turn += 1

            results[int(np.rint(reward * player)) + 1] += 1
            for j, example in enumerate(episode_examples):
                example[3] = reward if player == example[2] else -reward
                episode_examples[j] = [example[0], example[1].tolist(), example[3]]
            examples += episode_examples

        return examples, results

