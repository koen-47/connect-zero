from abc import ABC, abstractmethod


class Strategy:
    def __init__(self):
        pass

    @abstractmethod
    def calculate_move(self, board):
        pass
