import itertools

import unittest
import numpy as np
import pyomo.environ as pyo
import pyomo.core as pyo_core

from scipy.stats import multivariate_normal

from rare_pattern_detect.minlp_based import (
    MINLPModel,
    minlp_has_rare_pattern,
)
from rare_pattern_detect.patterns import PatternSpace, PatternSpaceType


class TestMINLPHasRarePattern(unittest.TestCase):

    # def test_PatternSpace_initialization(self):
    #     '''
    #     This test is used to make sure that these parameters are set correctly:
    #         - cutoff (min_area in 2D)
    #         - PatternSpaceType (type of pattern space. See Enum in patterns.py)
    #     '''
    #     min_area = 0.1
    #     pattern_space = PatternSpace(
    #         PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=min_area
    #     )
    #     assert (
    #         pattern_space.type is PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES
    #     ), "pattern_space type not set to defined type during initialization"
    #     assert (
    #         pattern_space.cutoff is min_area
    #     ), "cutoff not set to min area during initialization"

    # def test_MINLP_model_creation(self):
    #     '''
    #     This model makes sure that the following cari
    #     '''
    #     training_set = multivariate_normal.rvs(size=(10, 2))
    #     pattern_space = PatternSpace(
    #         PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=0.01
    #     )
    #     solver = MINLPModel(training_set, min_area=pattern_space.cutoff)
    #     # use unittest.asserterror
    #     assert solver.model is not None, "Minlp model is none after model creation"
    #     assert solver.model.pattern is not None
    #     assert solver.model.included is not None
    #     assert solver.model.obj is not None
    #     assert solver.model.interval_lengths is not None

    # @TODO: fix this
    # def test_zero_min_area_makes_everything_an_anomaly(self):
    #     """
    #     When a point to be classified lies outside of the training set
    #     and the min_area is set to zero, then f_hat is always zero
    #     and hence the point is anomalous.
    #     """
    #     training_set = np.array(
    #         [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
    #     )
    #     x = np.array([1.0, 1.0])
    #     min_areas = range(0, 4)
    #     expected_results = [True, True, True]
    #     min_areas = [0.0]
    #     mus = [0.0,0.1,1.0]

    #     results = [(
    #         min_area,
    #         mu,
    #         minlp_has_rare_pattern(
    #             x,
    #             training_set,
    #             PatternSpace(
    #                 PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, min_area
    #             ),
    #             mu,
    #             debugging_minlp_model=False
    #     ))for min_area, mu in itertools.product(min_areas, mus)]

    #     results = np.array(results, dtype=object)
    #     models_labels = results[:,2]
    #     labels = [l for _ , (_,l) in enumerate(models_labels)]

    #     assert labels == expected_results, [ "When min area=0 of the calculated pattern then \
    #         f_hat always satisfies the inequality (f(h|x,D) < mu). Hence all points should be classfied as anomalous"]

    # @TODO: fix this
    def test_zero_mu_and_min_area_bigger_than_0_makes_everything_normal(self):
        """
        When a point to be classified lies outside of the training set,
        mu is set to zero and the min_area is set to bigger than zero,
        then f_hat must be always bigger or equal to mu. Hence the point is not anomalous.
        testcase:
            take a gaussian distribution and check if a point that lies on the sides get labeled as anomaleous or not
            Using the Isolation Forest algorithm on such a point is classified as anomalous, given samples from a
            gaussian distribution as training set.
            However, using the RarePatternDetect (pac-rpad) we expect the algorithm to fail given that mu is zero
        """
        # training_set = multivariate_normal.rvs(size=(20, 2))[:10]
        training_set = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 2.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        )
        point_to_be_classified = np.array([1.5, 1.5])  # training_set[-1]
        results = []
        min_areas, mus = [3.9], [0.0]  # 0.1, 0.2,

        for min_area, mu in itertools.product(min_areas, mus):
            pattern_space = PatternSpace(
                PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, min_area
            )
            model, label = minlp_has_rare_pattern(
                point_to_be_classified,
                training_set,
                pattern_space,
                mu=mu,
                debugging_minlp_model=False,
            )

            results.append(label)

        print(
            "test_zero_mu_and_min_area_bigger_than_0_makes_everything_normal -> results: ",
            results,
        )
        assert results == [False], [
            "If mu = 0, min_area > 0 \
            then f_hat always dissatisfies the inequality (f(h|x,D) < mu). \
            Hence all points should be classfied as not anomalous"
        ]

        ## set mu different than zero to get the opposite result
        results = []
        min_areas, mus = [3.9], [0.6]  # 0.1, 0.2,

        for min_area, mu in itertools.product(min_areas, mus):
            pattern_space = PatternSpace(
                PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, min_area
            )
            model, label = minlp_has_rare_pattern(
                point_to_be_classified,
                training_set,
                pattern_space,
                mu=mu,
                debugging_minlp_model=False,
            )

            results.append(label)

        print(
            "test_zero_mu_and_min_area_bigger_than_0_makes_everything_normal -> results: ",
            results,
        )
        assert results == [True], [
            "If largest_bounding_area > min_area > 0 and mu > f_hat,  \
            then point should be classfied as anomalous"
        ]

    # def test_MINLP_classify_result_is_True(self):
    #     '''
    #     This test is used to make sure that when the min_area is small (0.1),
    #     the test point (1.0,1.0) is classified as anomalous when
    #     the training set is defined by the points that define a unit square.
    #     '''
    #     training_set =  np.array(
    #         [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    #     )
    #     point_to_be_classified = np.array([0.5,0.5])
    #     min_area = 0.1
    #     mu = 0.1
    #     pattern_space = PatternSpace(
    #         PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=min_area
    #     )
    #     model, label= minlp_has_rare_pattern(
    #             point_to_be_classified,
    #             training_set,
    #             pattern_space,
    #             mu,
    #             debugging_minlp_model=False,
    #     )
    #     assert model is not None
    #     assert label is not None
    #     assert label is True

    # def test_MINLP_classify_throws_exception(self):
    #     '''
    #     This test throws an exception because the min_area is equal to the largest bounding area.
    #     Thus, the minlp optimization problem is infeasible.
    #     '''
    #     training_set =  np.array(
    #         [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
    #     )
    #     point_to_be_classified = np.array([1.0,1.0])
    #     min_area = 4.0
    #     mu = 1.0
    #     pattern_space = PatternSpace(
    #         PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=min_area
    #     )
    #     _, _= minlp_has_rare_pattern(
    #             point_to_be_classified,
    #             training_set,
    #             pattern_space,
    #             mu,
    #             debugging_minlp_model=False,
    #     )
    #     self.assertRaises(ValueError, msg="expected error in test_MINLP_classify_throws_exception")

    # def test_MINLP_classify_does_not_throw_exception(self):
    #     '''
    #     This test is able to classify the point given that
    #     the min_area is not superior to the largest bounding area.
    #     '''
    #     training_set =  np.array(
    #         [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
    #     )
    #     point_to_be_classified = np.array([1.0,1.0])
    #     min_area = 3.8
    #     mu = 1.0
    #     pattern_space = PatternSpace(
    #         PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=min_area
    #     )
    #     model, label = minlp_has_rare_pattern(
    #             point_to_be_classified,
    #             training_set,
    #             pattern_space,
    #             mu,
    #             debugging_minlp_model=False,
    #     )

    #     assert label in [True,False], "Model not able to solve trivial case"

    # def test_MINLP_add_point_to_model(self):
    #     pass

    # def test_MINLP_extract_pattern(self):
    #     pass

    # def test_MINLP_extract_points_included_in_pattern(self):
    #     pass


if __name__ == "__main__":
    unittest.main()
