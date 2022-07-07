import numpy as np
import pybnb


class AxisAlignedHyperRectangle:
    def __init__(self, intervals: np.ndarray):
        assert intervals.shape[1] == 2, "Unexpected shape of argument"
        # self.dimension = intervals.shape[1]
        self.intervals = intervals
        self.area = self.calculate_area()

    def calculate_area(self):
        return np.prod(np.apply_along_axis(lambda i: i[1] - i[0], axis=1, arr=self.intervals))

    def update_interval(self, index: int, new_interval: np.ndarray) -> None:
        self.intervals[index, :] = new_interval

    def contains(self, point: np.ndarray) -> bool:
        ints = self.intervals
        return all((ints[:, 0] <= point.T) & (point.T <= ints[:, 1]))

    def copy(self):
        return AxisAlignedHyperRectangle(self.intervals)


class BranchAndBoundSolver(pybnb.Problem):
    def __init__(self, training_data: np.ndarray, point_to_be_classified: np.ndarray):
        """
        :param training_data: the training data as an ndarray of shape (N,d)
        :param point_to_be_classified: a single point as an ndarray of shape (1,d).
        """
        self.points = training_data
        self.pattern: AxisAlignedHyperRectangle = self.find_bounding_pattern(self.points)
        self.point_to_be_classified = point_to_be_classified
        self.trajectory = []

    def sense(self):
        return pybnb.minimize

    def objective(self):
        return np.float(len(self.points) / self.pattern.area)

    def bound(self):
        return np.float(1. / self.pattern.area)

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
            child_pattern = self.find_bounding_pattern(child_points)
            if child_pattern.contains(self.point_to_be_classified) and child_pattern.area > 0:
                child = pybnb.Node()
                child.state = (child_points, child_pattern, self.point_to_be_classified)
                yield child
            # then on the right
            child_points = np.delete(self.points, argsort[-1, i], 0)
            child_pattern = self.find_bounding_pattern(child_points)
            if child_pattern.contains(self.point_to_be_classified) and child_pattern.area > 0:
                child = pybnb.Node()
                child.state = (child_points, child_pattern, self.point_to_be_classified)
                yield child

    def find_bounding_pattern(self, points: np.ndarray) -> AxisAlignedHyperRectangle:
        intervals = np.sort(points, axis=0)[[0, -1], :]
        return AxisAlignedHyperRectangle(intervals.T)

    # Reporting method
    def notify_new_best_node(self, node, current):
        self.trajectory.append(node.state[1].intervals)

# %%
