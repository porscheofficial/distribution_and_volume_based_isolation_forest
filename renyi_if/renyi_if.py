from sklearn.ensemble._iforest import IsolationForest, _average_path_length
import numpy as np
from rare_pattern_detect.if_based import renyi_divergence


class RenyiIsolationForest(IsolationForest):
    def __init__(self, alpha: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def get_scores_per_estimator(self, X, subsample_features=False):
        """
        given a fitted tree and a set of samples, returns for every sample the depths of the point in the estimators
        :param X: array-like of shape (n_samples, n_features)
        :returns depths: array-like of shape (n_samples, n_estimators)
        """
        n_samples = X.shape[0]

        depths = np.zeros((n_samples, self.n_estimators), order="f")

        for i, (tree, features) in enumerate(
            zip(self.estimators_, self.estimators_features_)
        ):
            X_subset = X[:, features] if subsample_features else X

            leaves_index = tree.apply(X_subset)
            node_indicator = tree.decision_path(X_subset)
            n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

            depths[:, i] = (
                np.ravel(node_indicator.sum(axis=1))
                + _average_path_length(n_samples_leaf)
                - 1.0
            )
        denominator = len(self.estimators_) * _average_path_length([self.max_samples_])
        raw_scores = np.divide(
            depths, denominator, out=np.ones_like(depths), where=denominator != 0
        )
        return raw_scores

    def predict(self, X):
        decision_func = self.decision_function(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func <= 0] = -1
        return is_inlier

    def _compute_chunked_score_samples(self, X):
        n_estimators = self.n_estimators
        scores_per_estimator = self.get_scores_per_estimator(X)
        uniform = np.ones(n_estimators) / n_estimators

        return 2 ** (
            -np.exp(-renyi_divergence(uniform, scores_per_estimator, self.alpha))
            # / n_estimators
        )

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        return -self._compute_chunked_score_samples(
            X,
        )


#%%
