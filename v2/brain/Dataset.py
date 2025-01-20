import numpy as np


class Dataset:
    def __init__(self, data=None):
        if data is None:
            data = []
        self.data = data

    def add(self, state, probs, player, with_symmetry=False):
        self.data += [state, probs, player, None]
        if with_symmetry:
            self.data += [np.fliplr(state), np.flip(probs), player, None]

    def set_rewards(self, reward, player):
        for i, example in enumerate(self.data):
            example[3] = reward if player == example[2] else -reward
            self.data[i] = [example[0], example[1], example[3]]

        # temp = [reward * ((-1) ** (self.data[i, 2] != player)) for i in range(len(self.data))]
        # print(np.array_equal(np.array(self.data[:, 3]), np.array(temp)))

    def shuffle(self):
        np.random.shuffle(self.data)
