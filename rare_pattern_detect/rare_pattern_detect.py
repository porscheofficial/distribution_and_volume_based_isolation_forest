from rare_pattern_detect.patterns import PatternSpaceType
from rare_pattern_detect.minlp_based import minlp_has_rare_pattern

import numpy as np 
class RarePatternDetect:
    def __init__(self, delta, tau, epsilon, pattern_space):
        self.training_data = None
        self.delta = delta
        self.tau = tau
        self.epsilon = epsilon
        self.pattern_space = pattern_space

        if pattern_space.type == PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES:
            self.has_rare_pattern = minlp_has_rare_pattern
        else:
            self.has_rare_pattern = None

    def load_training_data(self, training_data):
        self.training_data = training_data

    def is_anomalous(self, x):
        _, pred = self.has_rare_pattern(
            x, 
            self.training_data, 
            self.pattern_space, 
            self.tau + self.epsilon / 2
        )