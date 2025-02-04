import numpy as np
import torch
from tqdm import tqdm

from brain.MCTS import MCTS
from game.Board import encode_board


class SelfPlay:
    """
    Class to handle the self-play phase of the AlphaZero algorithm.
    """
    def __init__(self, game, logger):
        """
        Constructor for the SelfPlay class.
        :param game: instance of Game object that controls the logic of the Connect Four game.
        :param logger: logger to record all results per iteration.
        """
        self.game = game
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger

    def play_episodes(self, model, n_episodes, temp_threshold=15):
        """
        Plays a specified number of episodes between two identical MCTS algorithms to generate training data.
        :param model: dual residual network to predict values and policies.
        :param n_episodes: number of episodes to play per iteration.
        :param temp_threshold: turn threshold at which the temperature changes from 1 to 0.
        :return:
        """
        examples = []
        results = [0] * 3

        # Start self-play loop.
        for i in tqdm(range(n_episodes), desc="Self-play"):
            reward, n_turn, player = 0, 0, 1
            board = self.game.get_initial_board()

            # Create MCTS instance with the specified parameters.
            mcts = MCTS(self.game, model, self.device, c_puct=2., dir_alpha=0.5)

            episode_examples = []

            # Keep playing while no end game condition has been reached.
            while reward == 0:
                self.logger.log(f"(Self-play) Episode {i + 1}. Turn {n_turn + 1}.", to_iteration=True)

                # Flip state to perspective of current player.
                state = self.game.get_canonical_form(board, player)

                # Compute best action after completing all MCTS simulations for this turn. If the turn number is
                # less than 15, then the temperature remains 1 (otherwise 0). Also computes the probabilities
                # associated with each action (used to train the policy network).
                action, probs = mcts.get_action_prob(state, temp=int(n_turn < temp_threshold), device=self.device)

                # Encode the board into a 3 x 6 x 7 tensor.
                encoded_state = encode_board(state, player)

                # Add tensor to training data and use vertical symmetry to double number of examples.
                episode_examples += [[encoded_state, probs, player, None]]
                episode_examples += [[np.flip(encoded_state, 2).tolist(), np.flip(probs), player, None]]

                # Play the computed action from MCTS and record the results for this turn.
                board, player = self.game.get_next_state(board, player, action)
                self.logger.log(f"Action: {action}. Policy: {np.round(np.array(probs), decimals=4)}", to_iteration=True)
                self.logger.log(self.game.display(board, color=False), to_iteration=True)

                # Compute reward by checking if game has been won by either player, is a draw, or is still playing.
                reward = self.game.get_game_ended(board, player)

                # Increment number of turns for this episode.
                n_turn += 1

            # Record result from the episode.
            results[int(np.rint(reward * player)) + 1] += 1

            # Iterate over the training data for this episode and set the reward based on the outcome of that game.
            for j, example in enumerate(episode_examples):
                example[3] = reward if player == example[2] else -reward
                episode_examples[j] = [example[0], example[1].tolist(), example[3]]
            examples += episode_examples

        return examples, results

