"""Test the renyi isolation forest installation."""

import numpy as np

from sklearn.metrics import roc_auc_score

from typing import Tuple, Callable

from renyi_isolation_forest.pac_based_renyi_isolation_forest import (
    PACBasedRenyiIsolationForest,
)
from renyi_isolation_forest.depth_based_renyi_isolation_forest import (
    DepthBasedRenyiIsolationForest,
)


def generate_dataset_by_norm(
    d: int,
    N: int,
    contamination: float,
    random_process: Callable[[int, int], np.ndarray] = np.random.randn,
    norm_order: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate hypersphere points.

    Parameters
    ----------
    d : int.
        number of features.

    N : int.
        number of samples.

    contamination: float between 0 and 1.0
        contamination to choose the outliers.

    random_process: distribution type
        uniform distribution in this case.

    norm_order: int
        specifies the norm to use.

    Returns
    -------
    dataset: of shape(N,d)
        contains the inliers and outliers.

    labels: of shape(n_outliers,d)
        contains the labels.

    """
    dataset = random_process(N, d)
    labels = np.zeros(N, dtype=np.int8)
    norms = np.linalg.norm(dataset, ord=norm_order, axis=1)
    cutoff_idx = N - round(contamination * N)
    labels[np.argsort(norms)[cutoff_idx:]] = 1
    return dataset, labels


depth = 8
alpha = 1.0
dimensions = 2
samples = 256
norm = 2
print("np.random.randn: ", type(np.random.randn))
data, anomaly = generate_dataset_by_norm(
    d=dimensions,
    N=samples,
    contamination=0.1,
    random_process=np.random.randn,
    norm_order=norm,
)


# Example with depth based renyi isolation forest
clf = DepthBasedRenyiIsolationForest()
clf.fit(X=data)
score = clf.decision_function(data, alpha)
roc_auc_score_depth = roc_auc_score(~anomaly, score)
if roc_auc_score_depth > 0.9:
    print(
        f"AUCROC score for depth based IF: {roc_auc_score_depth}. Installation successfull"
    )
else:
    print("installation failed for depth based IF. aucroc: ", roc_auc_score_depth)

# Example with distribution based renyi isolation forest
clf = PACBasedRenyiIsolationForest()
clf.fit(X=data)
score = clf.decision_function(data, alpha)
roc_auc_score_pac = roc_auc_score(~anomaly, score)

if roc_auc_score_pac > 0.9:
    print(
        f"AUCROC score for PAC based IF: {roc_auc_score_pac}. Installation successfull"
    )
else:
    print("installation failed for depth based IF. aucroc: ", roc_auc_score_pac)
