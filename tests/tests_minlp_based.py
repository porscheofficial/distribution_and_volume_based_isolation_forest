import itertools

import unittest
import numpy as np

from rare_pattern_detect.minlp_based import minlp_has_rare_pattern
from rare_pattern_detect.patterns import PatternSpace, PatternSpaceType


class TestMINLPHasRarePattern(unittest.TestCase):

    def test_zero_min_area_makes_everything_an_anomaly(self):
        """
        When a point to be classified lies outside of the training set and the min_area is set to zero, then f_hat
        is always zero and hence the point is anomalous.
        """
        training_set = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0]]
        )  # multivariate_normal.rvs(size=(10,2))
        x = np.array([[1.0, 1.0]])
        min_areas = np.range(0, 4)

        mus = np.range(0.1, 0.1, 1)
        for min_area, mu in itertools.product(min_areas, mus):
            pattern_space = PatternSpace(PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, min_area)
            result = minlp_has_rare_pattern(x, training_set, pattern_space, mu)
            self.assertTrue(result, "some explanation")

        y = np.array([[0, 2.0]])
        for min_area, mu in itertools.product(min_areas, mus):

            # f_hat will be area(h)

            pattern_space = PatternSpace(PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, min_area)
            result = minlp_has_rare_pattern(x, training_set, pattern_space, mu)
            self.assertTrue(result, "some explanation")
