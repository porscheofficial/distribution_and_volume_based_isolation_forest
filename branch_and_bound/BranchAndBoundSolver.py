from abc import ABC
from copy import deepcopy

import numpy as np
import pybnb

from patterns import AxisAlignedHyperRectangle, find_bounding_pattern


# def rankmin(x):
#     u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
#     csum = np.zeros_like(counts)
#     csum[1:] = counts[:-1].cumsum()
#     return csum[inv]

class BranchAndBoundSolver(pybnb.Problem, ABC):
    def __init__(self,
                 training_data: np.ndarray,
                 point_to_be_classified: np.ndarray,
                 min_area_factor: np.float):
        """
        :param training_data: the training data as an ndarray of shape (N,d)
        :param point_to_be_classified: a single point as an ndarray of shape (1,d).
        :param min_area_factor: the fraction of the bounding_area that any patter has to have.
        """
        self.points = training_data
        self.N, self.d = self.points.shape
        self.bounding_pattern: AxisAlignedHyperRectangle = find_bounding_pattern(self.points)
        self.bounding_area = self.bounding_pattern.area
        self.min_area_factor = min_area_factor
        self.min_area = self.bounding_area * self.min_area_factor
        self.point_to_be_classified = point_to_be_classified
        self.point_in_training = point_to_be_classified in training_data
        self.trajectory = []
        self.leading_index = 0
        self.counter_counter = 0
        self.argsort = np.argsort(self.points, axis=0)

    def sense(self):
        return pybnb.minimize

    def objective(self):
        # note that this is slightly inaccurate, namely when the point to be classified is in the pattern and sits
        # on the boundary.
        return np.float(
            np.count_nonzero(self.removed_points == False) * self.bounding_area / (self.pattern.area * self.N)
        )

    def save_state(self, node):
        node.state = (self.removed_points,
                      self.pattern,
                      self.move_counter,
                      self.leading_index)

    def load_state(self, node):
        (self.removed_points,
         self.pattern,
         self.move_counter,
         self.leading_index) = node.state

    # Reporting method
    def notify_new_best_node(self, node, current):
        self.trajectory.append(node.state[1].intervals)

    def atomic_branch(self, idx, sense):
        # TODO: consider edge case that we reached end of argsort
        # we don't need to worry about the counter moving out of bounds, because by construction no point can
        side = idx % 2
        i = (idx - side) // 2
        child_counter = deepcopy(self.move_counter)
        # before_argsort = rankmin(child_counter)

        # print(f"before {before_argsort}")

        match (sense, side):
            case ("in", 0) | ("out", 1):
                child_counter[i, side] += 1
            case ("out", 0) | ("in", 1):
                child_counter[i, side] -= 1
            case _:
                raise Exception("Branching sense not defined")
        # find the point that marks the new boundary
        # print(f"counter {child_counter}")
        new_outer_point_idx = self.argsort[child_counter[i, side], i]
        # update the pattern
        new_outer_value = self.points[new_outer_point_idx, i]
        # new_interval = deepcopy(self.pattern.intervals[i, :])
        # new_interval[side] = new_outer_value
        child_pattern = deepcopy(self.pattern)
        child_pattern.intervals[i, side] = new_outer_value
        # remove the boundary point from counting
        child_removed_points = deepcopy(self.removed_points)
        child_removed_points[new_outer_point_idx] = True
        self.counter_counter += 1
        if child_pattern.contains(self.point_to_be_classified) and child_pattern.area > self.min_area:
            child = pybnb.Node()
            child_leading_index = idx
            child.state = (child_removed_points, child_pattern, child_counter, child_leading_index)
            yield child


class BranchAndBoundTopDown(BranchAndBoundSolver):
    def __init__(self, training_data, point_to_be_classified, min_area_factor):
        super().__init__(training_data, point_to_be_classified, min_area_factor)
        # this counts the number of moves per dimension and hence is an integer-valued (d,2) matrix
        self.pattern = self.bounding_pattern
        self.move_counter = np.zeros((self.d, 2), dtype='int')
        self.move_counter[:, 1] = -1
        self.removed_points = np.zeros(len(self.points), dtype=bool)
        # we initially remove the boundary points from the bounding pattern.
        self.removed_points[self.argsort[0, :]] = True
        self.removed_points[self.argsort[-1, :]] = True

    def branch(self):
        """
        For the space of d-dimensional axis aligned hyper rectangles, branch will create [2d] children, one for each
        side of the existing pattern.
        """
        for idx in range(self.leading_index, 2 * self.d):
            # branch on the left side
            yield from self.atomic_branch(idx, "in")
            # then on the right
            yield from self.atomic_branch(idx, "in")

    def bound(self):
        return np.float(1. * self.bounding_area / (self.pattern.area * self.N)) if self.point_in_training else 0


def find_smallest_encompassing_pattern(points, point, min_area):
    distances = points - point  # For each point the difference to the corresponding point coordinate
    d = points.shape[1]
    move_counter = np.zeros((d, 2), dtype='int')
    for i in range(d):
        for j in range(2):
            if j == 0:
                condition = lambda x: np.argmin(x[x > 0])
            else:
                condition = lambda x: np.argmax(x[x <= 0])
            move_counter[i, j] = np.apply_along_axis(condition, 0, distances)
    pattern = AxisAlignedHyperRectangle(intervals=points[move_counter])
    return pattern, move_counter


class BranchAndBoundBottomUp(BranchAndBoundSolver):
    def __init__(self, training_data, point_to_be_classified, min_area_factor):
        super().__init__(training_data, point_to_be_classified, min_area_factor)
        self.pattern, self.move_counter = find_smallest_encompassing_pattern(self.points, self.point_to_be_classified, self.min_area)
        self.removed_points = np.zeros(len(self.points), dtype=bool)

    def branch(self):
        """
        For the space of d-dimensional axis aligned hyper rectangles, branch will create [2d] children, one for each
        side of the existing pattern.
        """
        for idx in range(self.leading_index, 2 * self.d):
            # branch on the left side
            yield from self.atomic_branch(idx, "out")
            # then on the right
            yield from self.atomic_branch(idx, "out")

    def bound(self):
        maximal_pattern = deepcopy(self.pattern)
        for idx in range(self.leading_index, self.d):
            side = idx % 2
            i = (idx - side) // 2
            maximal_pattern.intervals[i, side] = self.bounding_pattern.intervals[i, side]
        return np.float(
            np.count_nonzero(self.removed_points == False) * self.bounding_area / (maximal_pattern.area * self.N)
        )

# %%
