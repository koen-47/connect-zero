import unittest

from v2.experiment.Experiment import Experiment


class TestExperiment(unittest.TestCase):
    def test_experiment(self):
        experiment = Experiment("../models/saved/resnet_v5.pth")
        experiment.run(4, log_losses=True)
