import unittest

from experiment.Experiment import Experiment


class TestExperiment(unittest.TestCase):
    def test_experiment(self):
        experiment = Experiment("../models/recent/resnet_v4_128_5.pth")
        experiment.run(100, log_losses=True)
