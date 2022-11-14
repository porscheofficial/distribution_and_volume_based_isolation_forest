import itertools

import unittest
import numpy as np
import pyomo.environ as pyo

from scipy.stats import multivariate_normal

from rare_pattern_detect.minlp_based import (
    MINLPModel,
    minlp_has_rare_pattern,
)  # , adjust_largest_pattern_bounds
from rare_pattern_detect.patterns import PatternSpace, PatternSpaceType


class TestMINLPHasRarePattern(unittest.TestCase):

    def test_PatternSpace_initialization(self):
        min_area = 0.1
        pattern_space = PatternSpace(
            PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=min_area
        )
        assert (
            pattern_space.type is PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES
        ), "pattern_space type not set to defined type during initialization"
        assert (
            pattern_space.cutoff is min_area
        ), "cutoff not set to min area during initialization"

    def test_MINLP_model_creation(self):
        training_set = multivariate_normal.rvs(size=(10, 2))
        pattern_space = PatternSpace(
            PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=0.01
        )
        solver = MINLPModel(training_set, min_area=pattern_space.cutoff)
        assert solver.model is not None, "Minlp model is none after model creation"
        assert solver.model.pattern is not None
        assert solver.model.included is not None
        assert solver.model.obj is not None
        assert solver.model.interval_lengths is not None

    def test_zero_min_area_makes_everything_an_anomaly(self):
        """
        When a point to be classified lies outside of the training set and the min_area is set to zero, then f_hat
        is always zero and hence the point is anomalous.
        """
        training_set = np.array(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
        )
        x = np.array([1.0, 1.0])
        min_areas = range(0, 4)
        results = []
        min_areas = [0.0]
        mus = [
            0.0,
            0.1,
            1.0,
        ]  # 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #np.linspace(0.1, 1, 0.1)
        for min_area, mu in itertools.product(min_areas, mus):
            pattern_space = PatternSpace(
                PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, min_area
            )
            results.append(
                (
                    min_area,
                    mu,
                    minlp_has_rare_pattern(
                        x, 
                        training_set, 
                        pattern_space, 
                        mu, 
                        debugging_minlp_model=False
                    ),
                )
            )
        print("test_zero_min_area_makes_everything_an_anomaly -> results: ", results)
        np.testing.assert_array_equal(
            np.asarray(results)[:, 2],
            [True, True, True],
            [
                "When min area=0 of the calculated pattern then \
            f_hat always satisfies the inequality (f(h|x,D) < mu). \
            Hence all points should be classfied as anomalous"
            ],
        )

    def test_zero_mu_and_min_area_bigger_than_0_makes_everything_normal(self):
        """
        When a point to be classified lies outside of the training set,
        mu is set to zero and the min_area is set to bigger than zero,
        then f_hat must be always bigger than mu and
        hence the point is not anomalous.
        # testcase:
            # take a gaussian distribution and check if a point that lies on the sides get labeled as anomaleous or not
            # Using the Isolation Forest algorithm on such a point is classified as anomalous, given samples from a
            # gaussian distribution as training set.
            # However, using the RarePatternDetect (pac-rpad) we expect the algorithm to fail given that mu is zero
        """
        training_set = multivariate_normal.rvs(size=(20, 2))[:10]
        #print("training_set:", training_set)
        point_to_be_classified = training_set[-1]  # np.array([[2.0, 2.0]])
        #print("point_to_be_classified: ", point_to_be_classified)
        # y = np.array([[0, 2.0]])
        results = []
        min_areas, mus = [0.1, 0.2, 1.0], [0.0]
        expected_results = [False, False, False]
        for min_area, mu in itertools.product(min_areas, mus):
            pattern_space = PatternSpace(
                PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, min_area
            )
            result = (
                min_area,
                mu,
                minlp_has_rare_pattern(
                    point_to_be_classified,
                    training_set,
                    pattern_space,
                    mu,
                    debugging_minlp_model=False,
                ),
            )
            results.append(result)
        print("test_zero_mu_and_min_area_bigger_than_0 -> results: ", results)
        np.testing.assert_array_equal(
            np.asarray(results)[:, 2],
            [False, False, False],  # expected_results,
            [
                "When mu = 0, min_area > 0 \
            then f_hat always dissatisfies the inequality (f(h|x,D) < mu). \
            Hence all points should be classfied as not anomalous"
            ],
        )

    def test_MINLP_classify(self):
        training_set =  np.array(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
        )
        #print("training_set:", training_set)
        point_to_be_classified = np.array([1.0,1.0])  # np.array([[2.0, 2.0]])
        #print("point_to_be_classified: ", point_to_be_classified)
        min_area = 0.1
        mu = 0.1
        expected_results = [False]
        pattern_space = PatternSpace(
            PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=min_area
        )
        result = (
            min_area,
            mu,
            minlp_has_rare_pattern(
                point_to_be_classified,
                training_set,
                pattern_space,
                mu,
                debugging_minlp_model=True,
            ),
        )
        print("test_MINLP_classify -> result: ", np.asarray(result)[2])
        assert result is not None
        assert result[2] is True or False
        # assert result is False, "When mu > 0, min_area > 0 and the point to be classfied lies in the pattern \
        #     then f_hat always dissatisfies the inequality (f(h|x,D) < mu). \
        #     Hence all points should be classfied as not anomalous"
  

    # TODO: FIX CODE FIRST
    # def test_MINLP_model_creation_adjust_largest_pattern_bounds(self):
        # training_set = [[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]]
        # print("training_set:", training_set)
        # point_to_be_classified = np.array([[0.5, 0.5]])
        # result, results = [], []
        # # expected_results =  [False,False,False]

        # min_area = 0.2  # the smallest allowed area
        # N, d = training_set.shape
        # Nrange, drange = (range(x) for x in training_set.shape)
        # largest_bounding_area = np.array(
        #     [
        #         [np.min(training_set[:, 0]), np.min(training_set[:, 1])],
        #         [np.max(self.training_set[:, 0]), np.max(training_set[:, 1])],
        #     ]
        # )
        # model = pyo.ConcreteModel()

        # ## variables

        # # x is a 2d vector
        # model.d_dimension = pyo.Set(initialize=drange)
        # model.matrix = pyo.Set(initialize=model.d_dimension * range(2))
        # np.testing.assert_array_equal(
        #     np.asarray(results)[:, 2],
        #     [[0.0,0.0],[0.0,1.0]] # expected_results,
        #     [
        #         "Add something"
        #     ],
        # )

        # model.pattern = pyo.Var(model.matrix, bounds=_adjust_largest_pattern_bounds)
        # print(model.pattern)

    def test_MINLP_add_point_to_model(self):
        pass

    def test_MINLP_extract_pattern(self):
        pass

    def test_MINLP_extract_points_included_in_pattern(self):
        pass

    #     self.rare_pattern_detect = RarePatternDetect(self.training_set, min_area=self.min_area)
    #     result = None
    #     result = self.rare_pattern_detect.classify(self.point_to_be_classified)
    #     print("result: ", result)
    #     assert result is not None

if __name__ == "__main__":
    unittest.main()