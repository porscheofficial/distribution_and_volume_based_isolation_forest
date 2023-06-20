"""Contain a class that wraps around the IsolationForestWithMaxDepth."""

from sklearn.ensemble._iforest import _average_path_length
import numpy as np
from typing import Optional
from .utils import renyi_divergence, IsolationForestWithMaxDepth


class DepthBasedRenyiIsolationForest(IsolationForestWithMaxDepth):
    """

    Extends the IF algorithm with a renyi divergence to aggregate the scores.

    Instead of the average over the depths extracted from each tree,
    the algorithm can select different aggregation functions
    to calculate the anomaly score. Using different alpha values leads
    to different sensitivity to outliers.

    Ex:
        Given an small alpha value. For a point to be considered anomalous,
        The depth reached by the sample must be high for most of the trees.
        Given a high alpha value. For a point to be considered anomalous,
        it is enough to have one tree that isolates the samples at a high depth.
    """

    def __init__(self, **kwargs: str):
        super().__init__(**kwargs)

    def _set_oob_score(self, X: np.ndarray, y: Optional[np.ndarray]) -> None:
        raise NotImplementedError("OOB score not supported by iforest")

    def get_scores_per_estimator(
        self,
        X: np.ndarray,
        subsample_features: bool = False,
    ) -> np.ndarray:
        """
        Calculate the score for each estimator.

        Given a fitted tree and a set of samples, returns for every
        sample the depths of the point in the estimators.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            contains the features for each sample

        subsample_features: bool
            True to select a subset of the features to use by respective estimator.
            False to use the whole set of features.

        Returns
        -------
        depths: array-like of shape (n_samples, n_estimators)
            the individual depths reached by each sample for each tree of the forest

        """
        n_samples = X.shape[0]

        depths = np.zeros((n_samples, self.n_estimators), order="f")

        for i, (tree, features) in enumerate(
            zip(self.estimators_, self.estimators_features_)
        ):
            x_subset = X[:, features] if subsample_features else X

            leaves_index = tree.apply(x_subset)
            node_indicator = tree.decision_path(x_subset)
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

    def predict(self, X: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """
        Predict the scores.

        Given a forest of fitted trees and a set of samples, predict
        for a particular sample if it is an anomaly or not.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            contains the samples and their features

        alpha: float, has to be larger than zero
            this value is used to define the Renyi divergence

        Returns
        -------
        is_inlier: array-like of shape (n_samples)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.

        """
        decision_func = self.decision_function(X, alpha)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func <= 0] = -1

        return is_inlier

    def score_samples(self, X: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """
        Compute the anomaly score.

        Compute the score for a set of an input sample as the aggregated anomaly score
        of the trees in the forest using the Renyi divergences.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        alpha: float, has to be larger than zero
            this value is used to define the Renyi divergence

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.

        """
        return -self._compute_chunked_score_samples(X, alpha)

    def _compute_chunked_score_samples(
        self, X: np.ndarray, alpha: float = 0.0
    ) -> np.ndarray:
        n_estimators = self.n_estimators
        scores_per_estimator = self.get_scores_per_estimator(X)
        uniform = np.ones(n_estimators) / n_estimators
        return 2 ** (-np.exp(-renyi_divergence(uniform, scores_per_estimator, alpha)))


# %%
