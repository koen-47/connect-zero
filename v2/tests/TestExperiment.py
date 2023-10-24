import unittest

from v2.experiment.Experiment import Experiment


class TestExperiment(unittest.TestCase):
    def test_experiment(self):
        experiment = Experiment("../models/recent/resnet_128_8_71.pth")
        experiment.run(200, log_losses=True)

