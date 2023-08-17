import numpy as np


class LearningRateGetter:
    def __init__(self, multiplier):
        self.multiplier = multiplier
        self.num_evals = 0

    def reset(self):
        self.num_evals = 0

    def get_learning_rate(self):
        raise NotImplementedError


class ConstantLR(LearningRateGetter):
    def get_learning_rate(self):
        self.num_evals += 1
        return self.multiplier