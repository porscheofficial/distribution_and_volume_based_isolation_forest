from rare_pattern_detect.patterns import PatternSpaceType, PatternSpace
from rare_pattern_detect.minlp_based import minlp_has_rare_pattern

import numpy as np


class RarePatternDetect:
    def __init__(self, delta, tau, epsilon, pattern_space: PatternSpace):
        self.training_data = None
        self.testing_data = None
        self.delta = delta
        self.tau = tau
        self.epsilon = epsilon
        self.pattern_space = pattern_space

        if pattern_space.type == PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES:
            self.has_rare_pattern = minlp_has_rare_pattern
        else:
            self.has_rare_pattern = None

    def load_data(self, training_data, testing_data=None):
        self.training_data = training_data
        self.testing_data = testing_data
        N, d = training_data.shape
        # self.pattern_space.cutoff = self.pattern_space.calculate_coeff(
        #     epsilon=self.epsilon, delta=self.delta, N=N, d=d
        # )

    def is_anomalous(self, x):
        model, pred = self.has_rare_pattern(
            x,
            training_data=self.training_data,
            testing_data=self.testing_data,
            pattern_space=self.pattern_space,
            mu=self.tau + self.epsilon / 2,
        )
        return model, pred

    ## Added for ADBench
    def fit(self, training_data, testing_data=None):
        # print("testing_data: ", testing_data)
        # Here it returns false if the array is empty else it returns true
        if np.any(testing_data):
            print("testing_data not empty")
            self.load_data(training_data, testing_data)
        else:
            print("testing_data empty")
            self.load_data(training_data, None)

    ## Added for ADBench
    def predict_score(self, X_test):
        return self.is_anomalous(X_test)
