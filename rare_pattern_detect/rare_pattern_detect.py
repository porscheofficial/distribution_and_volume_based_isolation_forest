from minlp_based import has_rare_pattern


class RarePatternDetect:
    def __init__(self, delta, tau, epsilon, backend):
        self.training_data = None
        self.delta = delta,
        self.tau = tau,
        self.epsilon = epsilon
        if backend == "minlp":
            self.has_rare_pattern = has_rare_pattern
        else:
            self.has_rare_pattern = None

    def load_training_data(self, training_data):
        self.training_data = training_data

    def is_anomalous(self, x):
        return self.has_rare_pattern(
            x,
            self.training_data,
            self.tau + self.epsilon/2
        )