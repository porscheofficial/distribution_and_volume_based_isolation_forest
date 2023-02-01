from sklearn.ensemble import IsolationForest
import numpy as np


def calculate_bounding_pattern(X):
    """
    This returns the bounding pattern and returns it in the shape (d,2):
    """
    d = X.shape[1]
    result = np.zeros((d, 2), dtype=float)
    for i in range(d):
        result[i] = np.array([np.min(X[:, i]), np.max(X[:, i])])
    return result


class IFBasedRarePatternDetect(IsolationForest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f_hat_cache = None
        self.bounding_pattern = None

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)

        self.bounding_pattern = calculate_bounding_pattern(X)

        f_hat_cache = []
        for tree in self.estimators_:
            # this introduces a lot of overhead, this should ideally use only the number of leaves
            n = tree.tree_.node_count
            f_hat_cache.append(np.full((n,), fill_value=np.infty))

        self.f_hat_cache = f_hat_cache

    def score_samples(self, X):
        min_f_hats = self._get_pac_rpad_estimate(X)
        return min_f_hats

    def _get_pac_rpad_estimate(self, X):
        """

        :param X: {array-like} of shape (n_samples, n_features)
        :return: {array-like} of shape (n_samples, )
        """
        f_hats = self._get_f_hats(X)  # (n_samples, n_estimators)
        return np.min(f_hats, axis=1)

    def _get_f_hats(self, X):
        """
        :param X: {array-like} with shape (n_samples, n_features)
        :rtype: f_hats: {array-like} with shape (n_samples, n_estimators)
        """
        n_samples = X.shape[0]
        f_hats = np.empty((n_samples, self.n_estimators))
        for i, tree in enumerate(self.estimators_):

            # we get the node indices of the leaf nodes, one per sample.
            leaves_index = tree.apply(X)
            # we get the values of the corresponding cache
            cached_f_hat = self.f_hat_cache[i][leaves_index]  # (n_samples, )
            # we check which of them have already been calculated
            cache_is_calculated = cached_f_hat < np.infty  # (n_samples) : bool
            f_hats[cache_is_calculated, i] = cached_f_hat[cache_is_calculated]

            cache_not_calculated = ~cache_is_calculated
            uncached_X = X[cache_not_calculated]

            # the leaves_indices for which the cache is not calculated
            uncached_patterns = leaves_index[cache_not_calculated]
            if len(uncached_patterns) == 0:
                continue

            new_f_hats = self._calculate_f_hat(tree, uncached_X, uncached_patterns)  # (

            f_hats[cache_not_calculated, i] = new_f_hats
            self.f_hat_cache[i][uncached_patterns] = new_f_hats

        return f_hats

    def _calculate_f_hat(self, tree, X, leaves_index):

        areas = self._calculate_pattern_area_samples(tree, X, leaves_index)
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

        return n_samples_leaf / areas

    def _calculate_pattern_area_samples(self, tree, X, leaves_index):
        def area_from_pattern(pattern):
            return np.prod(pattern[:, 1] - pattern[:, 0])

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
