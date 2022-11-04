import pyomo.environ as pyo
import numpy as np


def has_rare_pattern(x, training_data, mu):
    min_area = 0.1  # @TODO: Replace with dynamic area calculation
    m = MINLPModel(training_data, min_area)
    s = m.classify(x)  # TODO: Parse solution output
    return s < mu


class MINLPModel:
    def __init__(self, training_set: np.array, min_area: float):
        self.training_set = training_set  # a N x d matrix
        self.min_area = min_area  # the smallest allowed area
        self.Nrange, self.drange = (range(x) for x in self.training_set.shape)
        self.model = self.create_model()

    def create_model(self):
        def pattern_area():
            return pyo.prod(model.interval_lengths[i] for i in self.drange)

        # define model
        model = pyo.ConcreteModel()

        # variables

        # x is a 2d vector
        # TODO: Set domain
        model.pattern = pyo.Var(
            range(2),
            self.drange,
            bounds=(np.min(self.training_set), np.max(self.training_set)),
        )

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
            expr=sum(model.included[i] for i in self.Nrange) / pattern_area()
        )

        ## constraints

        # pattern area needs to exceed min_area
        model.area_constraint = pyo.Constraint(expr=pattern_area() >= self.min_area)

        # this constraint enforces that points inside the pattern must be included (the converse needs not be constrained explicitly since including points outside
        # the pattern will increase the objective function unnecessarily.

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
                    >= model.pattern[1, i] + 1e-3
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
                model.interval_lengths[i] == (model.pattern[1, i] - model.pattern[0, i])
            )

        return model

    def add_point_to_model(self, point):
        # point to be classified lies in pattern
        # x[i] <= point[i] <= x[i + d], for all i
        self.model.point_constraint = pyo.ConstraintList()
        for i in self.drange:
            self.model.point_constraint.add(self.model.pattern[0, i] <= point[i])
            self.model.point_constraint.add(point[i] <= self.model.pattern[1, i])

    def classify(self, point_to_be_classified: np.array) -> bool:
        self.add_point_to_model(
            point_to_be_classified
        )  # point to be classified is a 1 x d array
        # return pyo.SolverFactory('gdpopt').solve(self.model, algorithm='LOA', mip_solver='glpk', nlp_solver='ipopt')
        return pyo.SolverFactory("mindtpy").solve(
            self.model, strategy="OA", mip_solver="glpk", nlp_solver="ipopt", tee=True
        )
