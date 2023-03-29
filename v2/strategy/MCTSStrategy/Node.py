class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.visits = 0
        self.score = 0
        self.children = None

    def is_leaf(self):
        return self.children is None or self.children == []

    def is_terminal(self):
        return self.children == []
