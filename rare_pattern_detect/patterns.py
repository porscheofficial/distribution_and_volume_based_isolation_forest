from enum import Enum
import math


class PatternSpaceType(Enum):
    AXIS_ALIGNED_HYPER_RECTANGLES = 1
    HALF_SPACES = 2


class PatternSpace:
    def __init__(self, type: PatternSpaceType, cutoff: float = 0):
        self.type = type
        self.cutoff = cutoff

    def calculate_coeff(self, **kwargs):
        if self.type == PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES:
            epsilon = kwargs["epsilon"]
            delta = kwargs["delta"]
            N = kwargs["N"]
            d = kwargs["d"]
            v = 2 * d
            min_area = 1 / N * math.log(d)  # TODO: FIll in details
            self.cutoff = min_area
