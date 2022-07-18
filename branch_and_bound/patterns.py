import numpy as np


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


def find_bounding_pattern(points: np.ndarray) -> AxisAlignedHyperRectangle:
    intervals = np.sort(points, axis=0)[[0, -1], :]
    return AxisAlignedHyperRectangle(intervals.T)
