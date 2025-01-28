import numpy as np
import torch

from v2.game.Board import encode_board


class MCTS:
    """
    Adapted from: https://github.com/suragnair/alpha-zero-general/tree/master
    My improvements include the use of nodes to form the MCTS process (to prevent potentially costly repeated
    state-action pair lookups) and the addition of Dirichlet noise to the policy vectors to add randomness (as denoted
    in the paper to address the exploitation-exploration trade-off).
    NOTE: Dirichlet noise is added during testing.
    """

    def __init__(self, game, model, device, num_sims=50, c_puct=1., dir_alpha=1., dir_e=0.25):
        self.game = game
        self.model = model.to(device)
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.nodes = {}
        self.n_actions = self.game.get_action_size()

        self.dir_alpha = dir_alpha
        self.dir_e = dir_e

    def get_action_prob(self, canonical_board, device, temp=0):
        self.model.eval()
        for i in range(self.num_sims):
            self.search(canonical_board, is_root=True, device=device)

        node_key = tuple(map(tuple, canonical_board))
        node = self.nodes[node_key]
        counts = [node.n_child_visits[a] for a in range(self.n_actions)]

        if temp == 0:
            probs = np.zeros(self.n_actions)
            probs[np.argmax(counts)] = 1
            best_action = np.random.choice(len(probs), p=probs)
            return best_action, probs

        counts = np.array([x ** (1. / temp) for x in counts])
        probs = counts / np.sum(counts)
        best_action = np.random.choice(len(probs), p=probs)
        return best_action, probs

    def search(self, board, is_root, device):
        node_key = tuple(map(tuple, board))
        if node_key not in self.nodes:
            self.nodes[node_key] = Node(board)
        node = self.nodes[node_key]

        if node.reward is None:
            node.reward = self.game.get_game_ended(board)
        if node.is_terminal():
            return -node.reward

        if node.policy is None:
            policy, value = self.__calculate_policy_value(board, device)
            node.valid_moves = self.game.get_valid_moves(board)
            node.policy = policy * node.valid_moves
            node.n_visits = 0
            return -value

        action = node.select(is_root, self.c_puct, self.dir_alpha, self.dir_e)
        next_board, next_player = self.game.get_next_state(board, 1, action)
        next_board = self.game.get_canonical_form(next_board, next_player)
        value = self.search(next_board, is_root=False, device=device)

        if node.q_values[action] is not None:
            node.q_values[action] = (node.n_child_visits[action] * node.q_values[action] + value) \
                                    / (node.n_child_visits[action] + 1)
            node.n_child_visits[action] += 1
        else:
            node.q_values[action] = value
            node.n_child_visits[action] = 1

        node.n_visits += 1
        return -value

    def __calculate_policy_value(self, board, device):
        board = encode_board(board, 1)
        tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
        tensor_state = tensor_state.unsqueeze(dim=0).to(device)
        policy, value = self.model(tensor_state)
        policy = policy.detach().cpu().numpy().flatten()
        policy = np.exp(policy) / np.sum(np.exp(policy), axis=0)
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

        self.n_visits = 0
        self.n_child_visits = [0] * 7
        self.q_values = [None] * 7

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
            return self.q_values[action] + c_puct * p * np.sqrt(self.n_visits) / (1 + self.n_child_visits[action])
        return c_puct * p * np.sqrt(self.n_visits + 1e-8)
