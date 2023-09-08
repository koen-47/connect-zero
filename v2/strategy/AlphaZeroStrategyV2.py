import math

import numpy as np
import torch


class AlphaZeroStrategyV2:
    def __init__(self, mcts):
        self.mcts = mcts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def calculate_move(self, canonical_board, player_id):
        pi = self.mcts.get_action_prob(canonical_board, device=self.device)
        action = np.random.choice(len(pi), p=pi)
        return action


class MCTS:
    """
    Adapted from: https://github.com/suragnair/alpha-zero-general/tree/master
    My improvements include the use of nodes to form the MCTS process (to prevent potentially costly repeated
    state-action pair lookups) and the addition of Dirichlet noise to the policy vectors to add randomness (as denoted
    in the paper to  the exploitation-exploration trade-off).
    NOTE: Dirichlet noise is added during testing.
    """

    def __init__(self, game, model, device, num_sims=250, c_puct=1.):
        self.game = game
        self.model = model.to(device)
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.nodes = {}
        self.n_actions = self.game.get_action_size()

        self.temp = 1
        self.dir_alpha = 1.
        self.dir_e = 0.25

    def get_action_prob(self, canonical_board, device):
        for i in range(self.num_sims):
            self.search(canonical_board, is_root=True, device=device)

        node_key = tuple(map(tuple, canonical_board))
        node = self.nodes[node_key]
        counts = [node.child_visits[a] for a in range(self.n_actions)]

        if self.temp == 0:
            probs = np.zeros(self.n_actions)
            probs[np.argmax(counts)] = 1
            return probs

        counts = [c ** (1. / self.temp) for c in counts]
        probs = np.exp(counts - np.max(counts))
        return probs / probs.sum(axis=0)

    def search(self, board, is_root, device):
        node_key = tuple(map(tuple, board))
        if node_key not in self.nodes:
            self.nodes[node_key] = Node(board)
        node = self.nodes[node_key]

        if node.reward is None:
            node.reward = self.game.get_game_ended(board, 1)
        if node.is_terminal():
            return -node.reward

        if node.policy is None:
            policy, value = self.__calculate_policy_value(board, device)
            node.valid_moves = self.game.get_valid_moves(board)
            node.policy = policy * node.valid_moves
            return -value

        action = node.select(is_root, self.c_puct, self.dir_alpha, self.dir_e)
        next_board, next_player = self.game.get_next_state(board, 1, action)
        next_board = self.game.get_canonical_form(next_board, next_player)
        value = self.search(next_board, is_root=False, device=device)

        if node.q_values[action] is not None:
            node.q_values[action] = (node.child_visits[action] * node.q_values[action] + value) \
                                    / (node.child_visits[action] + 1)
            node.child_visits[action] += 1
        else:
            node.q_values[action] = value
            node.child_visits[action] = 1

        node.n_visits += 1
        return -value

    def __calculate_policy_value(self, board, device):
        tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
        tensor_state = tensor_state.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
        policy, value = self.model(tensor_state)
        policy = policy.detach().cpu().numpy().flatten()
        value = value.detach().cpu().numpy().flatten()[0]
        return policy, value


class Node:
    def __init__(self, state, reward=None, valid_moves=None, policy=None):
        if valid_moves is None:
            valid_moves = []
        self.state = state
        self.reward = reward
        self.valid_moves = valid_moves
        self.policy = policy

        self.q_values = [None] * 7
        self.child_visits = [0] * 7
        self.n_visits = 0

    def is_terminal(self):
        return self.reward != 0

    def select(self, is_root, c_puct, dir_alpha, dir_e):
        policy = self.policy
        if is_root and dir_e > 0:
            noise = np.random.dirichlet([dir_alpha] * len(self.valid_moves))
            policy = (1 - dir_e) * policy + dir_e * noise

        max_ucb = -np.inf
        best_action = -1
        for action in np.where(self.valid_moves)[0]:
            ucb = self.__compute_ucb_score(action, policy[action], c_puct)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action
        return best_action

    def __compute_ucb_score(self, action, p, c_puct):
        if self.q_values[action] is not None:
            return self.q_values[action] + c_puct * p * math.sqrt(self.n_visits) / (1 + self.child_visits[action])
        return 0
