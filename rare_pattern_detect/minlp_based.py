import pyomo.environ as pyo
import numpy as np
from rare_pattern_detect.patterns import PatternSpace


def minlp_has_rare_pattern(x, training_data, pattern_space: PatternSpace, mu):
    min_area = pattern_space.cutoff  # @TODO: Replace with dynamic area calculation
    m = MINLPModel(training_data, min_area)
    s = m.classify(x)  # TODO: Parse solution output
    # print(" mu: ", mu)
    # print("s < mu: ",s < mu)
    return s <= mu


class MINLPModel:
    def __init__(self, training_set: np.array, min_area: float):
        self.training_set = training_set  # a N x d matrix
        self.min_area = min_area  # the smallest allowed area
        self.N, self.d = self.training_set.shape
        self.Nrange, self.drange = (range(x) for x in self.training_set.shape)
        self.largest_bounding_area = np.array(
            [
                [np.min(self.training_set[:, 0]), np.min(self.training_set[:, 1])],
                [np.max(self.training_set[:, 0]), np.max(self.training_set[:, 1])],
            ]
        )
        self.model = self.create_model()

    def create_model(self):
        def _pattern_area():
            return pyo.prod(model.interval_lengths[i] for i in self.drange)

        # define model
        model = pyo.ConcreteModel()

        ## variables

        # x is a 2d vector
        model.d_dimension = pyo.Set(initialize=self.drange)
        model.matrix = pyo.Set(initialize=model.d_dimension * range(2))

        # TODO: simplify with np.apply_along_axis
        def _adjust_largest_pattern_bounds(model, i, j):
            # print("i,j: ",i,j)
            if (i, j) == (0, 0):
                min_b = np.min(self.training_set[:, i])
                max_b = np.max(self.training_set[:, i])
            elif (i, j) == (0, 1):
                min_b = np.min(self.training_set[:, i])
                max_b = np.max(self.training_set[:, i])
            elif (i, j) == (1, 0):
                min_b = np.min(self.training_set[:, j])
                max_b = np.max(self.training_set[:, j])
            else:  # (1,1)
                min_b = np.min(self.training_set[:, j])
                max_b = np.max(self.training_set[:, j])
            return (min_b, max_b)

        model.pattern = pyo.Var(model.matrix, bounds=_adjust_largest_pattern_bounds)

        # y is a boolean vector of size N
        model.included = pyo.Var(self.Nrange, within=pyo.Binary, initialize=0)

        # auxiliary variables
        model.interval_lengths = pyo.Var(self.drange, within=pyo.NonNegativeReals)
        model.point_left_of_pattern = pyo.Var(
            self.Nrange, self.drange, within=pyo.Binary, initialize=0
        )
        model.point_right_of_pattern = pyo.Var(
            self.Nrange, self.drange, within=pyo.Binary, initialize=0
        )

        ## objective (minimised by default)
        model.obj = pyo.Objective(
            expr=sum(model.included[i] for i in self.Nrange) / _pattern_area(),
            sense=pyo.minimize,
        )

        ## constraints

        # pattern area needs to exceed min_area
        model.area_constraint = pyo.Constraint(expr=_pattern_area() >= self.min_area)

        # training points included in model.included lie within the pattern (NB: In principle we would need to ensure that points not included are also
        # not included in model.included. However, since including points outside the pattern increases the objective, this is covered.)

        model.include_constraint = pyo.ConstraintList()
        model.enforce_point_left_of_pattern = pyo.ConstraintList()
        model.enforce_point_right_of_pattern = pyo.ConstraintList()
        M = 100000
        for j in self.Nrange:
            for i in self.drange:
                # enforcing auxiliary variables are correct: point_left_of_pattern[j,i] is True iff the jth training point lies strictly outside the pattern in ith dimension, etc.
                model.enforce_point_left_of_pattern.add(
                    (model.point_left_of_pattern[j, i] * M + self.training_set[j, i])
                    >= model.pattern[0, i]
                )
                model.enforce_point_left_of_pattern.add(
                    self.training_set[j, i] + 1e-3
                    <= (
                        model.pattern[0, i]
                        + (1 - model.point_left_of_pattern[j, i]) * M
                    )
                )
                model.enforce_point_right_of_pattern.add(
                    self.training_set[j, i]
                    <= (model.pattern[1, i] + model.point_right_of_pattern[j, i] * M)
                )
                model.enforce_point_right_of_pattern.add(
                    (
                        (1 - model.point_right_of_pattern[j, i]) * M
                        + self.training_set[j, i]
                    )
                    >= (model.pattern[1, i] + 1e-3)
                )

            model.include_constraint.add(
                # key bit: this constraint enforces that the model.included for jth point can be set to 0 only if the point is not contained in the pattern (as witnessed by the fact
                # that the corresponding auxiliary variables are all 0)
                model.included[j]
                + sum(
                    model.point_right_of_pattern[j, i]
                    + model.point_left_of_pattern[j, i]
                    for i in self.drange
                )
                >= 1
            )

        # connect auxiliary variables: interval lengths are differences of pattern points
        model.interval_constraint = pyo.ConstraintList()
        for i in self.drange:
            model.interval_constraint.add(
                model.interval_lengths[i] == model.pattern[1, i] - model.pattern[0, i]
            )

        return model

    def add_point_to_model(self, point):
        # point to be classified lies in pattern
        point = point.squeeze()
        assert point.shape == (2,)
        self.model.point_constraint = pyo.ConstraintList()
        for i in self.drange:
            # x[i] <= point[i] <= x[i + d], for all i
            self.model.point_constraint.add(self.model.pattern[0, i] <= point[i])
            self.model.point_constraint.add(point[i] <= self.model.pattern[1, i])

    def extract_points_included_in_pattern(self):
        included_points = []
        for i in self.model.included:
            if np.round(self.model.included[i].value, 1) == 1.0:
                included_points.append(self.training_set[i])
        return np.array(included_points)

    def extract_pattern(self):
        intervals = np.zeros((2, 2), dtype=float)
        for _, j in enumerate(self.model.pattern):
            intervals[j] = self.model.pattern[j].value
        return intervals.T

    def classify(self, point_to_be_classified: np.array):
        self.add_point_to_model(
            point_to_be_classified
        )  # point to be classified is a 1 x d array
        _ = pyo.SolverFactory("mindtpy").solve(
            self.model,
            strategy="OA",
            mip_solver="glpk",
            nlp_solver="ipopt",
            tee=True,
        )
        return pyo.value(self.model.obj)
