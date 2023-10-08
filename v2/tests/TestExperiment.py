import unittest

from v2.experiment.Experiment import Experiment


class TestExperiment(unittest.TestCase):
    def test_experiment(self):
        experiment = Experiment("../models/saved/resnet_v7.pth")
        experiment.run(200, log_losses=True)
