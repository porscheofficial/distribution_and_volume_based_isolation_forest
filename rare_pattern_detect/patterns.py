from enum import Enum


class PatternSpaceType(Enum):
    AXIS_ALIGNED_HYPER_RECTANGLES = 1
    HALF_SPACES = 2


class PatternSpace:
    def __init__(self, type: PatternSpaceType, cutoff):
        self.type = type
        self.cutoff = cutoff if cutoff != None else self.calculate_coeff()
        print("cutoff (patterns.py): ", self.cutoff)

    # min area can be solved with wolfram alpha and then plugged in here
    # TODO: using simpy to solve the equation directly in python
    def calculate_coeff(self, **kwargs):
        if self.type == PatternSpaceType.AXIS_ALIGNED_HYPER_RECTANGLES:
            epsilon = kwargs["epsilon"]
            delta = kwargs["delta"]
            N = kwargs["N"]
            d = kwargs["d"]
            v = 2 * d
            raise Exception("Not implemented")
            # return sympy...
