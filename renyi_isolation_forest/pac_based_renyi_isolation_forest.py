"""Contain a class that wraps around the IsolationForestWithMaxDepth."""

from typing import Tuple
import numpy as np
from joblib import Parallel, delayed
from .utils import renyi_divergence, IsolationForestWithMaxDepth


class PACBasedRenyiIsolationForest(IsolationForestWithMaxDepth):
    """
    Extends the Isolation Forest algorithm.

    Instead of using a depth based approach to evaluate the testing points,
    this method works with a distribution based
    scoring function and Renyi divergences to aggregate the scores.
    """

    def __init__(self, padding=0.0, **kwargs):
        super().__init__(**kwargs)
        self.bounding_volume = None
        self.area_cache = None
        self.bounding_pattern = None
        self.padding = padding

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        super().fit(X, y, sample_weight)

        self.bounding_pattern = self._calculate_bounding_pattern(X)
        self.bounding_volume = _area_from_pattern(self.bounding_pattern)
        self.area_cache = self._calculate_forest_volumes()

        return self

    def _calculate_bounding_pattern(self, X):
        """
        Calculate the bounding pattern.

        The smallest bounding pattern is the one that encompasses
        all the training data points.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        result: ndarray of shape (d,2).
            bounding pattern

        """
        dimension = X.shape[1]
        result = np.zeros((dimension, 2), dtype=float)
        for i in range(dimension):
            result[i] = np.array(
                [np.min(X[:, i]) - self.padding, np.max(X[:, i]) + self.padding]
            )

        return result

    def predict(self, X, alpha=np.inf):
        """
        Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        alpha: float between 0 and infinity
            renyi value to modify the aggregation function

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

    def score_samples(self, X, alpha=np.inf):
        """
        Calculate the scores.

        Score is calculated as exp**-r_alpha(1/n, ps/us)/n, where r_alpha
        is the alpha renyi-divergence, us are the area probabilities, ps are
        the density samples and n is the number of estimators in the tree.
        Note that us and ps itself are positive and less than
        1 but not normalized to sum to 1.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        alpha: float between 0 and infinity
            renyi value to modify the aggregation function

        Returns
        -------
        scores: array-like of shape (n_samples, )
            calculated renyi scores

        """
        average_samples_in_leaf, f_hat = self.get_sample_distributions(X)
        n_estimators = average_samples_in_leaf.shape[1]
        uniform = np.full_like(average_samples_in_leaf, 1.0 / n_estimators)
        denominator = f_hat * n_estimators
        ratio = np.divide(
            average_samples_in_leaf,
            denominator,
            out=np.ones_like(average_samples_in_leaf),
            where=denominator != 0,
        )

        return -2 ** -(np.exp(-renyi_divergence(uniform, ratio, alpha)))

    def get_pac_rpad_estimate(self, X):
        """
        Compute the pac rpad scores.

        Parameters
        ----------
        X: {array-like} of shape (n_samples, n_features
            The input samples.

        Returns
        -------
            result: {array-like} of shape (n_samples)

        """
        return self.score_samples(X, alpha=np.inf)

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
            volumes[node_id] = _area_from_pattern(pattern)

            # If the left and right child of a node is
            # not the same we have a split node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children
            # and depth to `stack` so we can loop through them
            if is_split_node:
                pattern_left = pattern.copy()
                pattern_left[feature[node_id], 1] = threshold[node_id]
                stack.append((children_left[node_id], pattern_left))

                pattern_right = pattern.copy()
                pattern_right[feature[node_id], 0] = threshold[node_id]
                stack.append((children_right[node_id], pattern_right))

        return volumes

    def get_sample_distributions(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the density estimates and uniform density masses for the samples.

        Calculating the densities under the fitted tree.
        Uses a cache to avoid calculating the area several times.

        Parameters
        ----------
        X: {array-like} with shape (n_samples, n_features)
            The input samples.

        Returns
        -------
            two {array-like} with shape (n_samples, n_estimators)
            n_samples_in_leaf / n_samples us
            density estimates

        """
        result = Parallel(n_jobs=-1, backend="threading")(
            delayed(self._get_tree_samples)(tree, X, self.area_cache[i])
            for i, tree in enumerate(self.estimators_)
        )

        n_samples_in_leaf, areas = zip(*result)
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

        for i, _ in enumerate(X):
            pattern = self.bounding_pattern.copy()
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

            areas[i] = _area_from_pattern(pattern)

        return areas


def _area_from_pattern(pattern):
    return np.prod(pattern[:, 1] - pattern[:, 0])
