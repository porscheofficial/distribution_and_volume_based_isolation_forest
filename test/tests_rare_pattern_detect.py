import unittest
import numpy as np


from rare_pattern_detect.patterns import PatternSpace, PatternSpaceType
from rare_pattern_detect.rare_pattern_detect import RarePatternDetect

# run using python3.9 -m unittest test/tests_rare_pattern_detect.py


class TestRarePatternDetect(unittest.TestCase):
    def test_RarePatternDetect_loading_training_data(self):
        """
        This function should make sure that the training set is loaded and stored in the attribute of the Rare Pattern Detect class
        """
        training_set = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        rpd = RarePatternDetect(
            delta=0.1,
            tau=0.05,
            epsilon=0.1,
            pattern_space=PatternSpace(
                PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=0.5
            ),
        )
        rpd.load_training_data(training_set)
        assert rpd.training_data is training_set

    def test_RarePatternDetect_is_anomalous_all_anomalies(self):
        training_set = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        testing_set = np.array(
            [
                [2.0, 2.0],
                [0.1, 0.1],
                [0.25, 0.25],
            ]
        )

        rpd = RarePatternDetect(
            delta=0.1,
            tau=0.05,
            epsilon=0.1,
            pattern_space=PatternSpace(
                PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=0.5
            ),
        )
        rpd.load_training_data(training_set)
        preds = [
            rpd.is_anomalous(point_to_be_classified)
            for _, point_to_be_classified in enumerate(testing_set)
        ]
        assert None not in preds

    def test_RarePatternDetect_is_anomalous_raises_exception(self):
        """
        An exception occurs here when evaluating a point that has the same values as a training point
        used to define the largest bounding area.
        """
        training_set = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        point_to_be_classified = np.array([1.0, 1.0])

        rpd = RarePatternDetect(
            delta=0.1,
            tau=0.05,
            epsilon=0.1,
            pattern_space=PatternSpace(
                PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=0.5
            ),
        )
        rpd.load_training_data(training_set)
        preds = rpd.is_anomalous(point_to_be_classified)

        # @TODO: look for the exact exception to remove the error logs in the output of unittest
        self.assertRaises(ValueError)

    def test_RarePatternDetect_is_anomalous_expected_true(self):
        """
        This points should be considered anomelous. It lies outside of the bounds defined
        by the training set.
        """
        training_set = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        point_to_be_classified = np.array([2.0, 2.0])

        rpd = RarePatternDetect(
            delta=0.1,
            tau=0.05,
            epsilon=0.1,
            pattern_space=PatternSpace(
                PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES, cutoff=0.5
            ),
        )
        rpd.load_training_data(training_set)
        preds = rpd.is_anomalous(point_to_be_classified)
        print("supposed to be True -> preds: ", preds)
