from enum import Enum
import math

from sympy.abc import x
from sympy import solve_rational_inequalities, Poly

# from rare_pattern_detect.minlp_based import MIN_AREA

MIN_AREA = 1  # 0.00001 # 3.9 # 0.1


class PatternSpaceType(Enum):
    AXIS_ALIGNED_HYPER_RECTANGLES = 1
    HALF_SPACES = 2


class PatternSpace:
    def __init__(self, type: PatternSpaceType, cutoff):
        self.type = type
        self.cutoff = cutoff if cutoff != None else self.calculate_coeff()
        print("cutoff (min_area): ", self.cutoff)

    def calculate_coeff(self, **kwargs):
        if self.type == PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES:
            epsilon = kwargs["epsilon"]
            delta = kwargs["delta"]
            N = kwargs["N"]
            d = kwargs["d"]
            v = 2 * d
            # TODO: min area can be solved with wolfram alpha and then plugged in here
            # OR using simpy to solve the equation directly in python
            # min_area = math.sqrt((1 / N) * (256 / epsilon**2) * ( v * math.log(256 / epsilon**2) + math.log(8 / delta)))
            self.cutoff = 0.1  # MIN_AREA
