import numpy as np
import pybnb

from patterns import AxisAlignedHyperRectangle, find_bounding_pattern

class BranchAndBoundSolver(pybnb.Problem):
    def __init__(self, training_data: np.ndarray, point_to_be_classified: np.ndarray):
        """
        :param training_data: the training data as an ndarray of shape (N,d)
        :param point_to_be_classified: a single point as an ndarray of shape (1,d).
        """
        self.points = training_data
        self.N = len(training_data)
        self.pattern: AxisAlignedHyperRectangle = find_bounding_pattern(self.points)
        self.bounding_area = self.pattern.area
        self.point_to_be_classified = point_to_be_classified
        self.trajectory = []

    def sense(self):
        return pybnb.minimize

    def objective(self):
        return np.float(len(self.points * self.bounding_area)/ (self.pattern.area * self.N))

    def bound(self):
        return np.float(1. * self.bounding_area / (self.pattern.area * self.N))

    def save_state(self, node):
        node.state = (self.points,
                      self.pattern,
                      self.point_to_be_classified)

    def load_state(self, node):
        self.points, self.pattern, self.point_to_be_classified = node.state

    def branch(self):
        """
        For the space of d-dimensional axis aligned hyper rectangles, branch will create [2d] children, one for each
        side of the existing pattern.
        """
        argsort = np.argsort(self.points,
                             axis=0)
        for i, interval in enumerate(self.pattern.intervals):
            # branch on the left side
            child_points = np.delete(self.points, argsort[0, i], 0)
            child_pattern = find_bounding_pattern(child_points)
            # new_left_bound = self.points[argsort[1, i], i]
            # assert new_left_bound >= interval[0]
            # if new_left_bound < interval[1]:
            #     child_pattern = self.pattern.copy()
            #     child_pattern.update_interval(i, np.array([new_left_bound, interval[1]]))
            if child_pattern.contains(self.point_to_be_classified) and child_pattern.area > 0:
                child = pybnb.Node()
                child.state = (child_points, child_pattern, self.point_to_be_classified)
                yield child
            # then on the right
            child_points = np.delete(self.points, argsort[-1, i], 0)
            child_pattern = find_bounding_pattern(child_points)
            if child_pattern.contains(self.point_to_be_classified) and child_pattern.area > 0:
                child = pybnb.Node()
                child.state = (child_points, child_pattern, self.point_to_be_classified)
                yield child
            # new_right_bound = self.points[argsort[-2, i], i]
            # assert new_right_bound <= interval[1]
            # if new_right_bound > interval[0]:
            #     child_pattern = self.pattern.copy()
            #     child_pattern.update_interval(i, np.array([interval[0], new_right_bound]))
            #     assert child_pattern.area <= self.pattern.area
            #     child = pybnb.Node()
            #     child.state = (child_points, child_pattern, self.point_to_be_classified)
            #     yield child

    # Reporting method
    def notify_new_best_node(self, node, current):
        self.trajectory.append(node.state[1].intervals)

# %%
