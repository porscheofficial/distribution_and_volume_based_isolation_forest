import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from .utils import renyi_divergence


class PACBasedRenyiIsolationForest(IsolationForest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bounding_volume = None
        self.area_cache = None
        self.bounding_pattern = None

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)

        self.bounding_pattern = calculate_bounding_pattern(X)
        self.bounding_volume = area_from_pattern(self.bounding_pattern)

        self.area_cache = self._calculate_forest_volumes()
        return self

    def predict(self, X, alpha=np.inf):
        """
        Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """
        decision_func = self.decision_function(X, alpha)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    def decision_function(self, X, alpha: float):
        # following the convention to return the negated value

        scores = -self.pac_score_samples(X, alpha)

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # This is because it correspnds to the score of a point that whose raw score is 1. and which hence
            # behvaes just like the expected average.
            self.offset_ = 0.5

        else:
            # else, define offset_ wrt contamination parameter
            self.offset_ = np.percentile(scores, 100.0 * self.contamination)

        return scores - self.offset_

    def pac_score_samples(self, X, alpha=np.inf):
        """
        Calculates the scores as exp**-r_alpha(1/n, ps/us)/n, where r_alpha is the alpha renyi-divergence,
        us are the area probabilities, ps are the density samples and n is the number of estimators in the tree.
        Note that us and ps itself are positive and less than 1 but not normalized to sum to 1.
        :param X:
        :param alpha:
        :return: scores: array-like of shape (n_samples, )
        """
        ps, us = self._get_sample_distributions(X)
        n_estimators = ps.shape[1]
        uniform = np.full_like(ps, 1.0 / n_estimators)
        denominator = us * n_estimators
        ratio = np.divide(ps, denominator, out=np.ones_like(ps), where=denominator != 0)

        return 2 ** -(
            np.exp(-renyi_divergence(uniform, ratio, alpha))
        )  # / n_estimators

    def get_pac_rpad_estimate(self, X):
        """
        :param X: {array-like} of shape (n_samples, n_features)
        :return: {array-like} of shape (n_samples, )
        """
        return self.pac_score_samples(X, alpha=np.inf)

    def _calculate_forest_volumes(self):
        volumes = Parallel(n_jobs=-1, backend="threading")(
            delayed(self._calculate_tree_volumes)(tree) for tree in self.estimators_
        )
        return volumes

    def _calculate_tree_volumes(self, tree):
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        volumes = np.zeros(shape=n_nodes)
        stack = [
            (0, self.bounding_pattern)
        ]  # start with the root node id (0) and its area (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, pattern = stack.pop()
            volumes[node_id] = area_from_pattern(pattern)

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                pattern_left = pattern.copy()
                pattern_left[feature[node_id], 1] = threshold[node_id]
                stack.append((children_left[node_id], pattern_left))

                pattern_right = pattern.copy()
                pattern_right[feature[node_id], 0] = threshold[node_id]
                stack.append((children_right[node_id], pattern_right))

        return volumes

    def _get_sample_distributions(self, X) -> (NDArray, NDArray):
        """
        Calculate the density estimates and uniform density masses for the samples under the fitted tree.
        Uses a cache to avoid calculating the area several times.
        :param X: {array-like} with shape (n_samples, n_features)
        :rtype: n_samples_in_leaf, us: two {array-like} with shape (n_samples, n_estimators)
        """
        r = Parallel(n_jobs=-1, backend="threading")(
            delayed(self._get_tree_samples)(tree, X, self.area_cache[i])
            for i, tree in enumerate(self.estimators_)
        )

        n_samples_in_leaf, areas = zip(*r)
        n_samples_in_leaf = np.array(n_samples_in_leaf).T
        areas = np.array(areas).T
        n_samples = X.shape[0]
        return (
            n_samples_in_leaf / n_samples,
            areas / self.bounding_volume,
        )

    def _get_tree_samples(self, tree, X, area_cache):
        # we get the node indices of the leaf nodes, one per sample.
        leaves_index = tree.apply(X)
        samples_in_leaves = tree.tree_.n_node_samples[leaves_index]
        areas = area_cache[leaves_index]

        return samples_in_leaves, areas

    def _calculate_pattern_area_samples(self, tree, X, leaves_index):

        node_indicator = tree.decision_path(X)
        features = tree.tree_.feature
        thresholds = tree.tree_.threshold

        areas = np.empty((len(X),))

        # inefficient but should be sufficient to get an idea about the method.

        for i, x in enumerate(X):

            pattern = self.bounding_pattern.copy()  # (d,2)
            node_index = node_indicator.indices[
                node_indicator.indptr[i] : node_indicator.indptr[i + 1]
            ]

            for node_id in node_index:
                # continue to the next node if it is a leaf node
                if leaves_index[i] == node_id:
                    continue

                feature = features[node_id]
                threshold = thresholds[node_id]

                if X[i, feature] <= threshold:
                    pattern[feature, 1] = threshold
                else:
                    pattern[feature, 0] = threshold

            areas[i] = area_from_pattern(pattern)
        return areas


def calculate_bounding_pattern(X):
    """
    This returns the bounding pattern and returns it in the shape (d,2):
    """
    d = X.shape[1]
    result = np.zeros((d, 2), dtype=float)
    for i in range(d):
        result[i] = np.array([np.min(X[:, i]), np.max(X[:, i])])
    return result


def area_from_pattern(pattern):
    return np.prod(pattern[:, 1] - pattern[:, 0])
