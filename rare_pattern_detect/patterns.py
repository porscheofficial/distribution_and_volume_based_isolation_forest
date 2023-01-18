from enum import Enum
import math

from sympy.abc import x
from sympy import solve_rational_inequalities, Poly

# from rare_pattern_detect.minlp_based import MIN_AREA

# MIN_AREA = 1  # 0.00001 # 3.9 # 0.1


class PatternSpaceType(Enum):
    AXIS_ALIGNED_HYPER_RECTANGLES = 1
    HALF_SPACES = 2


class PatternSpace:
    def __init__(self, type: PatternSpaceType, cutoff):
        self.type = type
        self.cutoff = cutoff if cutoff != None else self.calculate_coeff()
        print("cutoff (patterns.py): ", self.cutoff)

    def calculate_coeff(self, **kwargs):
        if self.type == PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES:
            epsilon = kwargs["epsilon"]
            delta = kwargs["delta"]
            N = kwargs["N"]
            d = kwargs["d"]
            v = 2 * d
            # min area can be solved with wolfram alpha and then plugged in here
            # TODO: using simpy to solve the equation directly in python
            # self.cutoff = 6.3  # 0917
            # using 63.0917 for 100 points -> infeasibility detected in deactivate_trivial_constraints
            # Feasibility subproblem infeasible. This should never happen. -> all f_hats are zero
            # return self.cutoff
