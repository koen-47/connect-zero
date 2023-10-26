import unittest

from v2.experiment.Experiment import Experiment


class TestExperiment(unittest.TestCase):
    def test_experiment(self):
        experiment = Experiment("../models/recent/resnet_small_v2.pth")
        experiment.run(200, log_losses=True)

