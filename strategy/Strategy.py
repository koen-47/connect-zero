from abc import abstractmethod


class Strategy:
    def __init__(self):
        pass

    @abstractmethod
    def calculate_move(self, board, player_id):
        pass
