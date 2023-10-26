import unittest

import numpy as np
import torch

from v2.models.pytorch.DualResidualNetwork import DualResidualNetwork


class TestModel(unittest.TestCase):
    def test_value_prediction_1(self):
        board = [[ 0,  0, 0,  0,  0, 0, 0],
                 [ 0,  0, 0,  0,  0, 0, 0],
                 [ 0,  0, 0,  0,  0, 0, 0],
                 [ 0,  0, 0,  0,  0, 0, 0],
                 [ 0,  0, 0, -1,  0, 0, 0],
                 [-1, -1, 1,  1,  1, 0, 0]]

        device = torch.device("cpu")
        model = DualResidualNetwork(num_channels=512, num_res_blocks=5).to(device)
        model.load_state_dict(torch.load("../models/recent/resnet_small_v2.pth"))

        tensor_state = torch.tensor(np.array(board), dtype=torch.float32)
        tensor_state = tensor_state.unsqueeze(dim=0).unsqueeze(dim=0).to(device)
        _, value = model(tensor_state)
        print(value)