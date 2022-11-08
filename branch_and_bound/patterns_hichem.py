import numpy as np
import numpy.ma as ma
import math

class AxisAlignedHyperRectangle:
    def __init__(self, intervals: np.ndarray):
        assert intervals.shape[1] == 2, "Unexpected shape of argument"
        # self.dimension = intervals.shape[1]
        self.intervals = intervals
        self.area = self.calculate_area()

    def calculate_area(self):
        area =  np.prod(np.apply_along_axis(lambda i: i[1] - i[0], axis=1, arr=self.intervals))
        #print("area: ", area)
        assert area > 0, "Check intervals -> calculated area is zero or negative !"
        return area 

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
    #print(" ---- find_smallest_encompassing_pattern ----")
    distances = points - point  # For each point the difference to the corresponding point coordinate
    squared_distances = [x**2 + y**2 for _, (x,y) in enumerate(distances)]

    d = points.shape[1]
    move_counter = np.zeros((d, 2), dtype='int')
    intervals = np.zeros((d, 2))
    for j in range(2):
        #print(f" ----------- iteration: {j} ------------")
        if j == 0:
            # get indices of the points that have coordinates that maximizes the distance between random point and target 
            cond = lambda x: np.argmax(ma.masked_greater_equal(x, 0, copy=True)) # masked_less_equal
        else:
            # get indices of the points that have coordinates that minimizes the distance between random point and target 
            cond = lambda x: np.argmin(ma.masked_less_equal(x, 0, copy=True)) # masked_greater
        indices = np.apply_along_axis(cond, 0, distances)
        #print(f" indices: {indices} and chosen distances {distances[indices]}")
        move_counter[:, j] = indices
        values = points[indices]
        const = 1e-1
        if j == 0:
            # x = min_x
            intervals[0, 0] = values[0, 0] # - const
            # y = min_y
            intervals[1, 0] = values[1, 1] # - const
        else:
            # width = max_x
            intervals[0, 1] = values[0, 0] # - const #+
            # height = max_y
            intervals[1, 1] = values[1, 1] # - const #+

    pattern = AxisAlignedHyperRectangle(intervals=intervals)

    # TODO: The points used to generate the pattern must be outside of the pattern 
    for _, p in enumerate(points):
        if (p!=point).all() and pattern.contains(p):
            print("pattern.contains(point): ", p)
    #print("smallest encompassing pattern:", pattern.area) 
    #print("move_counter :", move_counter) 
    return pattern, move_counter


def find_smallest_encompassing_pattern_squared_distances(points, point):
    #print(" ---- find_smallest_encompassing_pattern ----")
    #print("points: ",points)

    # Checking if point is in points 
    boolean = np.isin(point , points)
    if True in boolean:
        points = np.delete(points, np.where(points == point)).reshape(-1,2)

    distances = points - point  # For each point the difference to the corresponding point coordinate
    squared_distances = [x**2 + y**2 for _, (x,y) in enumerate(distances)]

    d = points.shape[1]
    move_counter = np.zeros((d, 2), dtype='int')
    intervals = np.zeros((d, 2))
    for j in range(2):
        #print(f" ----------- iteration: {j} ------------")
        if j == 0:
            # get indices of the points that have coordinates that maximizes the distance between random point and target 
            index = np.argmin(squared_distances) # masked_less_equal
        else:
            # get indices of the points that have coordinates that minimizes the distance between random point and target 
            index = np.argmin(np.delete(squared_distances,index)) # masked_greater

        #print(f" indices: {index} and chosen distances {distances[index]}")
        move_counter[:, j] = index
        #print("move_counter: ",move_counter)
        values = points[index]
        #print(f" values: {values}")
        const = 1e-1
        if j == 0:
            # x = min_x
            intervals[0, 0] = values[0] # - const
            # y = min_y
            intervals[1, 0] = values[1] # - const
        else:
            # width = max_x
            intervals[0, 1] = values[0] # + const #abs(values[0])
            # height = max_y
            intervals[1, 1] = values[1] # + const #abs(values[1])
        #print(f" intervals {intervals}")

    pattern = AxisAlignedHyperRectangle(intervals=intervals)

    # TODO: The points used to generate the pattern must be outside of the pattern 
    for _, p in enumerate(points):
        if (p!=point).all() and pattern.contains(p):
            print("pattern.contains(point): ", p)
    #print("smallest encompassing pattern:", pattern.area) 
    #print("move_counter :", move_counter) 
    return pattern, move_counter
