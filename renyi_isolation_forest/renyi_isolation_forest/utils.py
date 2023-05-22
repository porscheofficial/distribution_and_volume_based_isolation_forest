import numpy as np
from numpy.typing import NDArray
from scipy.special import rel_entr
from sklearn.ensemble._iforest import (
    IsolationForest,
    check_random_state,
    tree_dtype,
    issparse,
)
from warnings import warn
import numbers


class IsolationForestWithMaxDepth(IsolationForest):
    def __init__(self, max_depth="auto", **kwargs):
        super().__init__(**kwargs)
        self.max_depth = max_depth

    def fit(self, X, y=None, sample_weight=None):
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
                    "max_samples (%s) is greater than the "
                    "total number of samples (%s). max_samples "
                    "will be set to n_samples for estimation."
                    % (self.max_samples, n_samples)
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


def renyi_divergence(p: NDArray, q: NDArray, alpha: float) -> float:
    """
    Calculates the alpha-renyi divergence (wrt base 2) between two discrete probability vectors of the same length.
    :param p: shape (n, d) where d is the dimension of the vector and n is a set of samples for which divergence
    is calculated.
    :param q: has to have same shape as p.
    :param alpha: float, has to be larger than zero
    :return: array like of shape (n, )
    """

    if alpha < 0:
        raise ValueError("`alpha` must be a non-negative real number")

    if alpha == 0:
        D_alpha = -np.log(np.where(p > 0, q, 0).sum(axis=1))
    elif alpha == 1:
        D_alpha = rel_entr(p, q).sum(axis=1)
    elif alpha == np.inf:
        # ratio = np.divide(p,dd q, out=np.ones_like(p), where=q != 0)
        D_alpha = np.log(np.max(p / q, axis=1))
    else:
        nominator = p**alpha
        denominator = q ** (alpha - 1)
        # ratio = np.divide(
        #     nominator, denominator, out=np.ones_like(nominator), where=denominator != 0
        # )
        D_alpha = 1 / (alpha - 1) * np.log((nominator / denominator).sum(axis=1))

    return D_alpha
