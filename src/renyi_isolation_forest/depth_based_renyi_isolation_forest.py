"""Contain a class that wraps around the IsolationForestWithMaxDepth."""

from sklearn.ensemble._iforest import _average_path_length
import numpy as np
from .utils import renyi_divergence, IsolationForestWithMaxDepth


class DepthBasedRenyiIsolationForest(IsolationForestWithMaxDepth):
    """

    Extends the Isolation Forest algorithm with a renyi divergence to aggregate the scores.
    
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def get_scores_per_estimator(
        self,
        X,
        subsample_features=False,
    ):
        """
        Calculate the score for each estimator.

        Given a fitted tree and a set of samples, returns for every
        sample the depths of the point in the estimators.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            contains the features for each sample

        subsample_features: bool
            True to select a subset of the features that were used by the respective estimator.
            False to use to the whole set of features.

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
            depths, denominator,
            out=np.ones_like(depths), where=denominator != 0
        )

        return raw_scores

    def predict(self, X, alpha=0):
        """
        Given a forest of fitted trees and a set of samples, predict for a particular sample if it is an anomaly or not.
        
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

    def decision_function(self, X, alpha=0):
        """
        Aggregate anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the aggregated anomaly score of the trees in the forest.
        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point.

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
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.

        """
        scores = self.score_samples(X, alpha)

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # This is because it correspnds to the score of a point that
            # whose raw score is 1. and which hence behvaes just like
            # the expected average.
            self.offset_ = 0.5

        else:
            # else, define offset_ wrt contamination parameter
            self.offset_ = np.percentile(scores, 100.0 * self.contamination)

        return scores - self.offset_

    def score_samples(self, X, alpha=0):
        """
        Compute the anomaly score of an input sample as the aggregated anomaly score of the trees in the forest using the Renyi divergences.

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

    def _compute_chunked_score_samples(self, X, alpha=0):
        n_estimators = self.n_estimators
        scores_per_estimator = self.get_scores_per_estimator(X)
        uniform = np.ones(n_estimators) / n_estimators

        return 2 ** (
            -np.exp(-renyi_divergence(uniform, scores_per_estimator, alpha))
        )

# %%
