import itertools

import unittest
import numpy as np

from scipy.stats import multivariate_normal

from rare_pattern_detect.minlp_based import MINLPModel, minlp_has_rare_pattern
from rare_pattern_detect.patterns import PatternSpace, PatternSpaceType


class TestMINLPHasRarePattern(unittest.TestCase):
    def test_PatternSpace_initialization(self):
        min_area = 0.1
        pattern_space = PatternSpace(
            PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=min_area
        )
        assert pattern_space.type is PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES
        assert pattern_space.cutoff is min_area

    def test_MINLP_model_creation(self):
        training_set = multivariate_normal.rvs(size=(10, 2))
        pattern_space = PatternSpace(
            PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=0.01
        )
        solver = MINLPModel(training_set, min_area=pattern_space.cutoff)
        assert solver.model is not None
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
        )  # multivariate_normal.rvs(size=(10,2))
        x = np.array([[1.0, 1.0]])
        min_areas = range(0, 4)
        results = []
        min_areas = [0.0]  # , 0.1, 1.0]
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
                    minlp_has_rare_pattern(x, training_set, pattern_space, mu),
                )
            )
        print("results: (min_area, mu, res)", results)
        np.testing.assert_array_equal(
            np.asarray(results)[:, 2],
            [True, True, True],
            ["If min area=0 then everything should become anomalous"],
        )

    # # TODO:
    # def test_zero_mu_and_min_area_bigger_than_0(self):
    #     """
    #     When a point to be classified lies outside of the training set,
    #     mu is set to zero and the min_area is set to bigger than zero,
    #     then f_hat must be always bigger than mu and
    #     hence the point is not anomalous.
    #     # testcase:
    #         # take a gaussian distribution and check if a point that lies on the sides get labels as anomaleous or not
    #         # Theoretically according to (IF) such a point should be classified as anomalous
    #         # but we expected the algorithm to fail given that mu is zero
    #     """
    #     y = np.array([[0, 2.0]])
    #     for min_area, mu in itertools.product(min_areas, mus):
    #         # f_hat will be area(h)
    #         pattern_space = PatternSpace(PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, min_area)
    #         result = minlp_has_rare_pattern(x, training_set, pattern_space, mu)
    #         self.assertTrue(result, "some explanation")

    # TODO: def test_MINLP_add_point_to_model(self):
    #    pass

    # TODO: def test_MINLP_extract_pattern(self):
    #     pass

    # TODO: def test_MINLP_extract_points_included_in_pattern(self):
    #     pass

    # TODO: def test_MINLP_adjust_largest_pattern_bounds(self):
    #     pass

    # TODO: def test_MINLP_classify(self):
    #     pass

    #     self.rare_pattern_detect = RarePatternDetect(self.training_set, min_area=self.min_area)
    #     result = None
    #     result = self.rare_pattern_detect.classify(self.point_to_be_classified)
    #     print("result: ", result)
    #     assert result is not None
