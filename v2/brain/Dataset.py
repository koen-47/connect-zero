import numpy as np


class Dataset:
    def __init__(self, data=None):
        if data is None:
            data = np.empty((0, 4), dtype=object)
        self.data = data

    def add(self, state, probs, player, with_symmetry=False):
        self.data = np.append(self.data, [[state, probs, player, None]], axis=0)
        if with_symmetry:
            self.data = np.append(self.data, [[np.fliplr(state), np.flip(probs), player, None]], axis=0)

    def set_rewards(self, reward, player):
        self.data[:, 3] = [reward if player == self.data[i, 2] else -reward for i in range(len(self.data))]
        # temp = [reward * ((-1) ** (self.data[i, 2] != player)) for i in range(len(self.data))]
        # print(np.array_equal(np.array(self.data[:, 3]), np.array(temp)))

    def shuffle(self):
        np.random.shuffle(self.data)
