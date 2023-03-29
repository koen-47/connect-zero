import math
import random
from abc import ABC

# from game.Game import Game
from v2.strategy.Strategy import Strategy
from v2.strategy.MCTSStrategy.Node import Node
from v2.strategy.MCTSStrategy.Tree import Tree
from v2.strategy.RandomStrategy import RandomStrategy


class MCTSStrategy:
    def __init__(self, rollout_limit):
        super().__init__()
        self.rollout_limit = rollout_limit

    def move(self, game, player_id):
        tree = Tree()

        for _ in range(self.rollout_limit):
            self.__simulate_game(game.clone(), tree, player_id)

        return max(tree.root.children, key=lambda c: c.visits).action[1]

    def __simulate_game(self, game, tree, player_id):
        current_node = tree.root
        nodes_to_update = [current_node]

        while not current_node.is_leaf():
            current_node = self.__get_child_with_highest_ucb(current_node)
            player_id, col = current_node.action
            game.move(player_id, col)
            nodes_to_update.append(current_node)

        if not game.is_over():
            next_player = game.get_next_player()
            self.__add_children(current_node, game.board, next_player.id)

            best_new_child_node = self.__get_child_with_highest_ucb(current_node)
            game.move(next_player.id, best_new_child_node.action[1])

            random_strategy = RandomStrategy()
            while not game.is_over():
                next_player = game.get_next_player()
                random_move = random_strategy.calculate_move(game.board)
                game.move(next_player.id, random_move)

        win = game.winner == player_id
        draw = game.winner is None
        score = 1 if win else 0 if draw else -1

        for node_to_update in nodes_to_update:
            player_for_node = node_to_update.action[0] if node_to_update.action else None
            node_score = score
            if player_for_node != player_id:
                node_score *= -1
            node_to_update.visits += 1
            node_to_update.score += node_score

    def __add_children(self, parent_node, board, player_id):
        assert parent_node.children is None
        moves = board.get_valid_moves()

        children = []
        for move in moves:
            child_node = Node(parent_node, (player_id, move))
            children.append(child_node)

        parent_node.children = children

    def __get_child_with_highest_ucb(self, node):
        max_ucb = -math.inf
        max_children = []
        for child in node.children:
            child_ucb = self.__ucb(child)
            if child_ucb > max_ucb:
                max_children = [child]
                max_ucb = child_ucb

            elif self.__ucb(child) == max_ucb:
                max_children.append(child)

        return random.choice(max_children)

    def __ucb(self, node):
        if node.visits == 0:
            return math.inf
        return node.score / node.visits + 2 * (math.log(node.parent.visits) / node.visits) ** 0.5

    def get_name(self):
        return 'MC_' + str(self.rollout_limit)

    def game_over(self, reward):
        pass

    def save(self):
        pass


