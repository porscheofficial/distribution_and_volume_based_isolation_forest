"""Test the renyi dist isolation forest installation."""
from collections import defaultdict
from functools import partial
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import seaborn as sns


from renyi_isolation_forest.pac_based_renyi_isolation_forest import (
    PACBasedRenyiIsolationForest,
)
from renyi_isolation_forest.depth_based_renyi_isolation_forest import (
    DepthBasedRenyiIsolationForest,
)


def evaluate_clf(cls, data, labels, alpha, **kwargs):
    """
    Instantiate and fit estimator.

    Parameters
    ----------
    cls : depth or distribution based isolation forest models.
        The model.

    data : {array-like, sparse matrix} of shape (n_samples, n_features).
        The input samples.

    labels : of shape (n_samples).
        labels.

    alpha : float between 0 and infinity.
        renyi value to modify the aggregation function.

    Returns
    -------
    clf : object.
        Fitted estimator.

    roc_auc_score : float.
        calculated roc auc score.

    """
    clf = cls(**kwargs)
    clf.fit(data, data)
    return clf, roc_auc_score(labels, clf.decision_function(data, alpha))


def sample_hypersphere_points(N, d):
    """
    Generate hypersphere points.

    Parameters
    ----------
    N : int.
        number of samples.

    d : int.
        number of features.

    Returns
    -------
    unit_coords: of shape(N,d)
        normalized coordinated according to the specified norm.

    """
    aux = np.random.randn(N, d)
    lengths = np.linalg.norm(aux, ord=2, axis=1).reshape(-1, 1)
    unit_coords = aux / lengths
    return unit_coords


def generate_screening_dataset(d, N, contamination, random_process, radius, sigma):
    """
    Generate hypersphere points.

    Parameters
    ----------
    N: int.
        number of samples.

    d: int.
        number of features.

    contamination: float between 0 and 1.0
        contamination to choose the outliers.

    random_process: distribution type
        uniform distribution in this case.

    radius: float
        radius of the hypersphere.

    sigma: float
        used to select inliers

    Returns
    -------
    data: of shape(N,d)
        contains the inliers and outliers.

    anomaly: of shape(n_outliers,d)
        contains the outliers.

    """
    outlier_count = round(contamination * N)
    inlier_count = N - outlier_count
    radii = np.random.randn(inlier_count) * sigma + radius * d
    inliers = radii.reshape(-1, 1) * sample_hypersphere_points(inlier_count, d)
    outliers = random_process(outlier_count, d)
    data = np.vstack([inliers, outliers])
    anomaly = np.array([i >= inlier_count for i in range(N)])
    return data, anomaly


def run_experiment(d_lim, data_generation_process, alpha=0):
    """
    Generate the data, fit the estimator, run the prediction and evalute the results.

    Parameters
    ----------
    d_lim : int
        Integer that represents the dimension of the dataset. The number of features

    data_generation_process : function
        function responsible for the data generation process

    alpha : float between 0 and infinity
        renyi value to modify the aggregation function

    Returns
    -------
    clfs : dict object
        contains the fitted estimators.

    results : pandas Dataframe
        Dataframe containing the results

    """
    names = ["DepthBased", "AreaBased"]
    results = pd.DataFrame(columns=names)
    clfs = {name: defaultdict(int) for name in names}
    for d in range(1, d_lim):
        data, anomaly = data_generation_process(d=d)
        for clf, name, kwargs in [
            (DepthBasedRenyiIsolationForest, "DepthBased", {"max_depth": d**d}),
            (
                PACBasedRenyiIsolationForest,
                "AreaBased",
                {"padding": 0.1, "max_depth": d**d},
            ),
        ]:
            clfs[name][d], results.loc[d, name] = evaluate_clf(
                clf, data, ~anomaly, alpha, **kwargs
            )
    return clfs, results


generate_screening = partial(
    generate_screening_dataset,
    d=1,
    N=25600,
    contamination=0.001,
    random_process=np.random.randn,
    radius=2.0,
    sigma=0.1,
)
clfs, results = run_experiment(10, generate_screening, 2)
test_plot = sns.lineplot(results)
fig = test_plot.get_figure()
fig.savefig("images/test-out.png")
