from numpy._typing import NDArray
from sklearn.ensemble import IsolationForest
import numpy as np
from scipy.special import rel_entr


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


def renyi_divergence(p: NDArray, q: NDArray, alpha: float) -> float:
    """
    Calculates the alpha-renyi divergence (wrt base 2) between two discrete probability vectors of the same length.
    :param p: shape (n, d) where d is the dimension of the vector and n is a set of samples for which divergence
    is calculated.
    :param q: has to have same shape as p.
    :param alpha: float, has to be larger than zero
    :return: array like of shape (n, )
    """

    if p.shape != q.shape:
        raise ValueError("Input arrays need to have same shape")

    if alpha < 0:
        raise ValueError("`alpha` must be a non-negative real number")

    if alpha == 0:
        D_alpha = -np.log(np.where(p > 0, q, 0).sum(axis=1))
    elif alpha == 1:
        D_alpha = rel_entr(p, q).sum(axis=1)
    elif alpha == np.inf:
        D_alpha = np.log(np.max(p / q, axis=1))
    else:
        D_alpha = (
            1 / (alpha - 1) * np.log(((p**alpha) / (q ** (alpha - 1))).sum(axis=1))
        )

    return D_alpha


class IFBasedRarePatternDetect(IsolationForest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bounding_volume = None
        self.area_cache = None
        self.bounding_pattern = None

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)

        self.bounding_pattern = calculate_bounding_pattern(X)
        self.bounding_volume = area_from_pattern(self.bounding_pattern)

        area_cache = []
        for tree in self.estimators_:
            # this introduces a lot of overhead, this should ideally use only the number of leaves.
            # For a proper implementation, the area should be calculated during fitting the IF and stored in
            # a custom tree_ class. This would mean prediction speed similar to the original IF.
            n = tree.tree_.node_count
            area_cache.append(np.full((n,), fill_value=np.infty))

        self.area_cache = area_cache

    def get_if_scores(self, X):
        # creating an alias to the original function for conceptual clarity
        return self.score_samples(X)

    def pac_score_samples(self, X, alpha=np.inf):
        """
        Calculates the scores as exp**-r_alpha(1/n, ps/us)/n, where r_alpha is the alpha renyi-divergence,
        us are the area probabilities, ps are the density samples and n is the number of estimators in the tree. Note that us and ps itself
        are positive and less than 1 but not normalized to sum to 1.
        :param X:
        :param alpha:
        :return: scores: array-like of shape (n_samples, )
        """
        ps, us = self._get_sample_distributions(X)
        n_estimators = ps.shape[1]
        uniform = np.full_like(ps, 1.0 / n_estimators)

        return np.exp(-renyi_divergence(uniform, ps / us, alpha)) / n_estimators

    def get_pac_rpad_estimate(self, X):
        """
        :param X: {array-like} of shape (n_samples, n_features)
        :return: {array-like} of shape (n_samples, )
        """
        return self.score_samples(X, alpha=np.inf)

    def _get_sample_distributions(self, X, normalize=True) -> (NDArray, NDArray):
        """
        Calculate the density estimates and uniform density masses for the samples under the fitted tree.
        Uses a cache to avoid calculating the area several times.
        :param X: {array-like} with shape (n_samples, n_features)
        :rtype: ps, us: two {array-like} with shape (n_samples, n_estimators)
        """
        n_samples = X.shape[0]
        samples_in_leaves = np.empty((n_samples, self.n_estimators))
        areas = np.empty((n_samples, self.n_estimators))
        for i, tree in enumerate(self.estimators_):

            # we get the node indices of the leaf nodes, one per sample.
            leaves_index = tree.apply(X)
            samples_in_leaves[:, i] = tree.tree_.n_node_samples[leaves_index]

            # we get the values of the corresponding cache
            cached_area = self.area_cache[i][leaves_index]  # (n_samples, )
            # we check which of them have already been calculated
            cache_is_calculated = cached_area < np.infty  # (n_samples) : bool
            areas[cache_is_calculated, i] = cached_area[cache_is_calculated]

            cache_not_calculated = np.logical_not(cache_is_calculated)
            uncached_X = X[cache_not_calculated]

            # the leaves_indices for which the cache is not calculated
            uncached_patterns = leaves_index[cache_not_calculated]
            if len(uncached_patterns) == 0:
                continue

            new_areas = self._calculate_pattern_area_samples(
                tree, uncached_X, leaves_index
            )
            areas[cache_not_calculated, i] = new_areas
            self.area_cache[i][uncached_patterns] = new_areas

        if normalize:
            return samples_in_leaves / n_samples, areas / self.bounding_volume

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
