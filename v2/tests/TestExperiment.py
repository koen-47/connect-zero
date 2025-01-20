import unittest

from v2.experiment.Experiment import Experiment


class TestExperiment(unittest.TestCase):
    def test_experiment(self):
        experiment = Experiment("../models/recent/resnet_v2_64_4.pth")
        experiment.run(100, log_losses=True)
