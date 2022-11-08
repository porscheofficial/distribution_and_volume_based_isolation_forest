import numpy as np
import numpy.ma as ma


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


def find_smallest_encompassing_pattern(points, point):
    distances = points - point  # For each point the difference to the corresponding point coordinate
    d = points.shape[1]
    move_counter = np.zeros((d, 2), dtype='int')
    intervals = np.zeros((d, 2))
    for j in range(2):
        if j == 0:
            cond = lambda x: np.argmax(ma.masked_less_equal(x, 0, 0))
        else:
            cond = lambda x: np.argmin(ma.masked_greater(x, 0, 0))
        indices = np.apply_along_axis(cond, 0, distances)
        move_counter[:, j] = indices
        values = points[indices]
        intervals[:, j] = values[:, j]

    pattern = AxisAlignedHyperRectangle(intervals=intervals)

    return pattern, move_counter