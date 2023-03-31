import numpy as np
from numpy.typing import NDArray
from scipy.special import rel_entr


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
