from enum import Enum


class PatternSpaceType(Enum):
    AXIS_ALIGNED_HYPER_RECTANGLES = 1
    HALF_SPACES = 2


class PatternSpace:
    def __init__(self, type: PatternSpaceType, cutoff: float = 0):
        self.type = type
        self.cutoff = cutoff
