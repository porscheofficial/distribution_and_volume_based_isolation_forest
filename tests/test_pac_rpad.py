import unittest
import numpy as np

# import sys

from rare_pattern_detect.patterns import PatternSpace, PatternSpaceType

# sys.path.append('../')

from rare_pattern_detect.rare_pattern_detect import RarePatternDetect


class TestRarePatternDetect(unittest.TestCase):

    def setUp(self):
        self.training_set = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]]
        )  # multivariate_normal.rvs(size=(10,2))
        self.point_to_be_classified = np.array([[0.5, 0.5]])
        self.min_area = 0
        self.N, self.d = self.training_set.shape
        self.largest_bounding_area = np.array(
            [
                [np.min(self.training_set[:, 0]), np.min(self.training_set[:, 1])],
                [np.max(self.training_set[:, 0]), np.max(self.training_set[:, 1])]
            ]
        )

        self.pattern_space = PatternSpace(PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, self.min_area)

        self.rare_pattern_detect = RarePatternDetect(0, 0, 0, self.pattern_space)
        self.rare_pattern_detect.load_training_data(self.training_set)

    def tearDown(self):
        pass

    def test_RarePatternDetect_initialization(self):
        self.model = self.rare_pattern_detect.is_anomalous(self.point_to_be_classified)
        assert self.model is not None
        assert self.model.pattern is not None
        assert self.model.included is not None
        assert self.model.obj is not None

    def test_RarePatternDetect_classify(self):
        result = None
        result = self.rare_pattern_detect.is_anomalous(self.point_to_be_classified)
        assert result is not None

    # TODO:
    # write test for RarePatternDetect
    # add_point_to_model
    # largest_bounding_area
    # _adjust_pattern_bounds
