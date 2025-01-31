import numpy as np
import torch

from game.Board import encode_board


class MCTS:
    """
    Class to handle the MCTS algorithm.
    Adapted from: https://github.com/suragnair/alpha-zero-general/tree/master
    My improvements include the use of nodes to form the MCTS process (to prevent potentially costly repeated
    state-action pair lookups) and the addition of Dirichlet noise to the policy vectors to add randomness (as denoted
    in the paper).
    """

    def __init__(self, game, model, device, num_sims=100, c_puct=1., dir_alpha=1., dir_e=0.25):
        """
        Constructor for the MCTS algorithm.
        :param game: instance of Game object that controls the logic of the Connect Four game.
        :param model: residual network model.
        :param device: flag to indicate which device PyTorch will run on (CUDA or CPU).
        :param num_sims: number of MCTS simulations to perform per turn.
        :param c_puct: hyperparameter that controls the degree to explore moves (whether to put more stock into the
        policy or value estimates).
        :param dir_alpha: alpha value of the Dirichlet distribution noise that is added.
        :param dir_e: parameter to indicate the amount of noise added to the policy priors.
        """
        self.game = game
        self.model = model.to(device)
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.dir_alpha = dir_alpha
        self.dir_e = dir_e

        # Dictionary to hold nodes in the MCTS tree.
        self.nodes = {}

        # Size of action space.
        self.n_actions = self.game.get_action_size()

    def get_action_prob(self, canonical_board, device, temp=0):
        """
        Computes the action probabilities for a given state.
        :param canonical_board: state of the board (always from the perspective of player 1)
        :param device: flag to indicate which device PyTorch will run on (CUDA or CPU).
        :param temp: temperature variable for probability smoothing.
        :return: list of probabilities (indexed by action).
        """

        # Starts MCTS simulations from specified state.
        self.model.eval()
        for i in range(self.num_sims):
            self.search(canonical_board, is_root=True, device=device)

        # Get root node.
        node_key = tuple(map(tuple, canonical_board))
        node = self.nodes[node_key]

        # Compute number of visits of child nodes.
        counts = [node.n_child_visits[a] for a in range(self.n_actions)]

        # Compute deterministic policy for that state (if temperature == 0)
        if temp == 0:
            probs = np.zeros(self.n_actions)
            probs[np.argmax(counts)] = 1
            best_action = np.random.choice(len(probs), p=probs)
            return best_action, probs

        # Compute stochastic policy for that state (if temperature != 0)
        counts = np.array([x ** (1. / temp) for x in counts])
        probs = counts / np.sum(counts)
        best_action = np.random.choice(len(probs), p=probs)
        return best_action, probs

    def search(self, board, is_root, device):
        """
        Performs a single simulation of MCTS for the specified board.
        :param board: state of the board (root node)
        :param is_root: flag to denote if the current node is the root node (used for Dirichlet noise).
        :param device: flag to indicate which device PyTorch will run on (CUDA or CPU).
        :return: the opposite value/reward for the next node that has the opponent's perspective.
        """

        # Get node corresponding to board and add it if it's not in the tree.
        node_key = tuple(map(tuple, board))
        if node_key not in self.nodes:
            self.nodes[node_key] = Node(board)
        node = self.nodes[node_key]

        # Compute reward if node is a leaf node.
        if node.reward is None:
            node.reward = self.game.get_game_ended(board, 1)
        if node.is_terminal():
            return -node.reward

        # Compute policy of the node.
        if node.policy is None:
            policy, value = self.__calculate_policy_value(board, device)
            node.valid_moves = self.game.get_valid_moves(board)
            node.policy = policy * node.valid_moves
            node.n_visits = 0
            return -value

        # Select best action according to UCB.
        action = node.select(is_root, self.c_puct, self.dir_alpha, self.dir_e)

        # Perform action.
        next_board, next_player = self.game.get_next_state(board, 1, action)
        next_board = self.game.get_canonical_form(next_board, next_player)

        # Perform same process on the new state.
        value = self.search(next_board, is_root=False, device=device)

        # Update running average of the q-values of that node.
        if node.q_values[action] is not None:
            node.q_values[action] = (node.n_child_visits[action] * node.q_values[action] + value) \
                                    / (node.n_child_visits[action] + 1)
            node.n_child_visits[action] += 1
        # Initialize q-value
        else:
            node.q_values[action] = value
            node.n_child_visits[action] = 1

        node.n_visits += 1
        return -value

    def __calculate_policy_value(self, board, device):
        """
        Compute the policy and value of the specified state using the residual network.
        :param board: state of the board.
        :param device: flag to indicate which device PyTorch will run on (CUDA or CPU).
        :return: tuple consisting of policy (softmaxed) and value.
        """
        # Encode board and convert to tensor.
        board = encode_board(board, 1)
        tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
        tensor_state = tensor_state.unsqueeze(dim=0).to(device)

        # Compute policy and value
        policy, value = self.model(tensor_state)
        policy = policy.detach().cpu().numpy().flatten()

        # Normalize the policy.
        policy = np.exp(policy) / np.sum(np.exp(policy), axis=0)

        value = value.detach().cpu().numpy().flatten()[0]
        return policy, value


class Node:
    def __init__(self, state, reward=None, valid_moves=None, policy=None):
        """
        Constructor for node.
        :param state: state of the board.
        :param reward: computed reward of the state.
        :param valid_moves: list of valid moves that can be played for the state.
        :param policy: policy for the state.
        """
        if valid_moves is None:
            valid_moves = []
        self.state = state
        self.reward = reward
        self.valid_moves = valid_moves
        self.policy = policy

        # Number of visits to this node
        self.n_visits = 0

        # List containing the number of visits for each of this node's child nodes
        self.n_child_visits = [0] * 7

        # List containing the computed q-values for each action for this node.
        self.q_values = [None] * 7

    def is_terminal(self):
        """
        Check if the node is terminal (leaf node).
        :return: True if it is a leaf node, False otherwise.
        """
        return self.reward != 0

    def select(self, is_root, c_puct, dir_alpha, dir_e):
        """
        Selects the best action based on the best UCB.
        :param is_root: flag to denote if the node from which the action is selected is the root node.
        :param c_puct: hyperparameter that controls the degree to explore moves (whether to put more stock into the
        policy or value estimates).
        :param dir_alpha: alpha value of the Dirichlet distribution noise that is added.
        :param dir_e: parameter to indicate the amount of noise added to the policy priors
        :return: best action based on the computed UCB.
        """

        policy = self.policy

        # Add Dirichlet noise if the node is the root node.
        if is_root and dir_e > 0:
            noise = np.random.dirichlet([dir_alpha] * len(self.valid_moves))
            policy = (1 - dir_e) * policy + dir_e * noise

        # Compute UCB for each (valid) action and keep track of the best action with the highest UCB.
        max_ucb = -np.inf
        best_action = -1
        for action in np.where(self.valid_moves)[0]:
            ucb = self.__compute_ucb_score(action, policy[action], c_puct)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action

        # Return best action.
        return best_action

    def __compute_ucb_score(self, action, p, c_puct):
        """
        Compute the UCB score of the specified action.
        :param action: action to compute the UCB for.
        :param p: probability from the policy for this action.
        :param c_puct: hyperparameter that controls the degree to explore moves (whether to put more stock into the
        policy or value estimates).
        :return: UCB score of the specified action.
        """
        if self.q_values[action] is not None:
            return self.q_values[action] + c_puct * p * np.sqrt(self.n_visits) / (1 + self.n_child_visits[action])
        return c_puct * p * np.sqrt(self.n_visits + 1e-8)
