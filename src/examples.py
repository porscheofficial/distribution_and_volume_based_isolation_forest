"""Test the renyi isolation forest installation."""

import numpy as np
import numpy as np

from sklearn.metrics import roc_auc_score

from renyi_isolation_forest.pac_based_renyi_isolation_forest import PACBasedRenyiIsolationForest
from renyi_isolation_forest.depth_based_renyi_isolation_forest import DepthBasedRenyiIsolationForest


def generate_dataset_by_norm(d, N, contamination, random_process = np.random.randn, norm_order=2):
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
    data: of shape(N,d)
        contains the inliers and outliers.
    
    anomaly: of shape(n_outliers,d)
        contains the labels.

    """
    data = random_process(N, d)
    anomaly = np.zeros(N, dtype=np.int8)
    norms = np.linalg.norm(data, ord=norm_order,  axis=1)
    cutoff_idx = N - round(contamination*N)
    anomaly[np.argsort(norms)[cutoff_idx:]] = 1
    return data, anomaly

depth = 8 
alpha = 1.0
dimensions = 2
samples = 256
contamination = 0.1 
norm = 2 

data, anomaly = generate_dataset_by_norm(d=dimensions, 
                                         N=samples, 
                                         contamination=contamination, 
                                         random_process=np.random.randn, 
                                         norm_order=norm) 


# Example with depth based renyi isolation forest
clf = DepthBasedRenyiIsolationForest()
clf.fit(X=data)
score = -clf.decision_function(data, alpha)
roc_auc_score_depth = roc_auc_score(anomaly, score)
print("AUCROC Score for Depth based:", roc_auc_score_depth)

# Example with distribution based renyi isolation forest
clf = PACBasedRenyiIsolationForest()
clf.fit(X=data)
score = -clf.decision_function(data, alpha)
roc_auc_score_pac = roc_auc_score(anomaly, score)
print("AUCROC Score for PAC based:", roc_auc_score_pac)