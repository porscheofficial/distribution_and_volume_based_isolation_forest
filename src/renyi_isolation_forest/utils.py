"""Contains helper functions."""

import numbers
from warnings import warn
import numpy as np
from scipy.special import rel_entr
from sklearn.ensemble._iforest import (
    IsolationForest,
    check_random_state,
    tree_dtype,
    issparse,
)


class IsolationForestWithMaxDepth(IsolationForest):
    """

    Wrapper class around the original Isolation Forest class.

    It is an extension that allows setting a max depth for the trees of the forest.

    Parameters
    ----------
    max_depth : 'auto' or int, default='auto'
        the max depth that can be reached by each tree in the forest 
        when fitting the data
            - If 'auto', then max_depth is equal to 
                int(np.ceil(np.log2(max(max_samples, 2)))).
            - If 'int' then max_depth is equal to the specified integer.

    """

    def __init__(self, max_depth="auto", **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth
        self.offset_ = None
        self.max_samples_ = None

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def fit(self, X, y=None, sample_weight=None) -> IsolationForest:
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator.

        """
        self._validate_params()
        X = self._validate_data(X, accept_sparse=["csc"], dtype=tree_dtype)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            max_samples = min(256, n_samples)

        elif isinstance(self.max_samples, numbers.Integral):
            if self.max_samples > n_samples:
                warn(
                    f"max_samples {self.max_samples} is greater than the "
                    "total number of samples {n_samples}. max_samples "
                    "will be set to n_samples for estimation."
                )
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # max_samples is float
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples

        if self.max_depth == "auto":
            max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        else:
            max_depth = self.max_depth

        super()._fit(
            X,
            y,
            max_samples,
            max_depth=max_depth,
            sample_weight=sample_weight,
            check_input=False,
        )

        return self

    def decision_function(self, X, alpha: float=0):
        """
        Aggregate anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the aggregated anomaly score of the trees in the forest.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        alpha: float between 0 and infinity
            renyi value to modify the aggregation function

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.

        """
        # following the convention to return the negated value

        scores = -self.score_samples(X, alpha)

        if self.contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # This is because it correspnds to the score of a point that whose
            # raw score is 1. and which hence
            # behvaes just like the expected average.
            self.offset_ = 0.5

        else:
            # else, define offset_ wrt contamination parameter
            self.offset_ = np.percentile(scores, 100.0 * self.contamination)

        return scores - self.offset_

def renyi_divergence(p_array: np.ndarray, q_array: np.ndarray, alpha: float) -> float:
    """

    Calculate the alpha-renyi divergence.

    The alpha-renyi divergence (wrt base 2) is calculated between
    two discrete probability vectors of the same length.

    Parameters
    ----------
    p_array: ndarray of shape shape (n, d)
        where d is the dimension of the vector
        and n is a set of samples for which divergence is calculated.
        probability vector

    q_array:  ndarray of shape shape (n, d)
        where d is the dimension of the vector
        and n is a set of samples for which divergence is calculated.
        probability vector

    alpha: float, has to be larger than zero
            this value is used to define the Renyi divergence

    Returns
    -------
    d_alpha: array like of shape (n, )
        Calculate the renyi divergences for each probability
        between the two discrete probability vectors

    """
    if alpha < 0:
        raise ValueError("`alpha` must be a non-negative real number")

    if alpha == 0:
        d_alpha = -np.log(np.where(p_array > 0, q_array, 0).sum(axis=1))
    elif alpha == 1:
        d_alpha = rel_entr(p_array, q_array).sum(axis=1)
    elif alpha == np.inf:
        d_alpha = np.log(np.max(p_array / q_array, axis=1))
    else:
        nominator = p_array**alpha
        denominator = q_array ** (alpha - 1)
        d_alpha = 1 / (alpha - 1) * np.log((nominator / denominator).sum(axis=1))

    return d_alpha
