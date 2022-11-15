import unittest
import numpy as np


from rare_pattern_detect.patterns import PatternSpace, PatternSpaceType
from rare_pattern_detect.rare_pattern_detect import RarePatternDetect


class TestRarePatternDetect(unittest.TestCase):
    def test_RarePatternDetect_loading_training_data(self):
        """
        This function should make sure that the training set is loaded and stored in the attribute of the Rare Pattern Detect class
        """
        training_set = np.array(
            [
                [0.0, 0.0], 
                [1.0, 0.0], 
                [0.0, 1.0], 
                [1.0, 1.0]
            ]
        )

        pattern_space = PatternSpace(
            PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, 
            cutoff=0
        )

        rare_pattern_detect = RarePatternDetect(0, 0, 0, pattern_space)
        rare_pattern_detect.load_training_data(training_set)
        assert rare_pattern_detect.training_data is training_set

    def test_RarePatternDetect_is_anomalous(self):
        pass
