import pyomo.environ as pyo
import numpy as np
from rare_pattern_detect.patterns import PatternSpace, MIN_AREA


def minlp_has_rare_pattern(
    x, training_data, pattern_space: PatternSpace, mu, debugging_minlp_model=False
):
    min_area = MIN_AREA  # pattern_space.cutoff
    model = MINLPModel(training_data, min_area)

    # Checking if point is included in the largest bounding area defined by the training set
    if contains(x, model.largest_bounding_area):
        solution = model.classify(x, tee=debugging_minlp_model)

        # Parse solution output
        if solution is not None:
            # If the minlp pyomo_model was feasible and a solution was found
            # then we return the pyomo_model and the label that contains
            # if the point is anomalous or not (bool)
            res = (model, solution <= mu)
        else:
            # If for some reasons the pyomo_model encountered an error
            # while trying to solve the minlp pyomo_model
            print("Error when classifying a point: ", x, model.largest_bounding_area)
            res = (None, None)
    else:
        print("point to be classified outside of the limits: anomaly")
        # no need to solve the minlp pyomo_model for this point.
        # Since the point lies outside of the largest point area, then it must be an anomaly (True)
        res = (None, True)

    return res


def contains(point: np.ndarray, largest_bounding_area) -> bool:
    contained = all(
        (largest_bounding_area.T[0, :] <= point.T)
        & (point.T <= largest_bounding_area.T[1, :])
    )
    return contained


class MINLPModel:
    def __init__(self, training_set: np.array, min_area: float):
        # !! This should never happen -> f_hat is zero -> everything anomaleous"
        # -> A test was added to test for this case
        # assert min_area != 0.0, "min_area is zero
        self.training_set = training_set  # a N x d matrix
        self.min_area = min_area  # the smallest allowed area
        self.N, self.d = self.training_set.shape
        self.Nrange, self.drange = (range(x) for x in self.training_set.shape)
        self.largest_bounding_area = self.calculate_largest_bounding_area()
        self.pyomo_model = self.create_pyomo_model()
        self.minimized_f_hats = np.zeros(self.N, float)

    def create_pyomo_model(self):
        def _pattern_area():
            return pyo.prod(pyomo_model.interval_lengths[i] for i in self.drange)

        # define pyomo_model
        pyomo_model = pyo.ConcreteModel()

        # variables

        # x is a 2d vectorl

        pyomo_model.pattern = pyo.Var(
            range(2), self.drange, within=pyo.Reals  # self.drange , self.drange #
        )  # , bounds=adjust_largest_pattern_bounds)

        # y is a boolean vector of size N
        pyomo_model.included = pyo.Var(self.Nrange, within=pyo.Binary, initialize=1)

        # auxiliary variables

        def bounding_area_func(model, i):
            return float(
                self.largest_bounding_area[i, 1] - self.largest_bounding_area[i, 0]
            )

        pyomo_model.interval_lengths = pyo.Var(
            self.drange,
            within=pyo.NonNegativeReals,
            initialize=bounding_area_func,
        )
        pyomo_model.point_left_of_pattern = pyo.Var(
            self.Nrange, self.drange, within=pyo.Binary, initialize=0
        )
        pyomo_model.point_right_of_pattern = pyo.Var(
            self.Nrange, self.drange, within=pyo.Binary, initialize=0
        )

        ## objective (minimised by default)
        pyomo_model.obj = pyo.Objective(
            expr=sum(pyomo_model.included[i] for i in self.Nrange) / _pattern_area(),
            sense=pyo.minimize,
        )

        ## constraints

        # pattern area needs to exceed min_area
        pyomo_model.area_constraint = pyo.Constraint(
            expr=_pattern_area() >= self.min_area
        )

        # training points included in pyomo_model.included lie within the pattern (NB: In principle we would need to ensure that points not included are also
        # not included in pyomo_model.included. However, since including points outside the pattern increases the objective, this is covered.)

        pyomo_model.include_constraint = pyo.ConstraintList()
        pyomo_model.enforce_point_left_of_pattern = pyo.ConstraintList()
        pyomo_model.enforce_point_right_of_pattern = pyo.ConstraintList()
        M = 100000
        for j in self.Nrange:
            for i in self.drange:
                # enforcing auxiliary variables are correct: point_left_of_pattern[j,i] is True iff the jth training
                # point lies strictly left of the pattern in ith dimension, etc.

                # if point is strictly left of pattern, indicator needs to be set to True
                pyomo_model.enforce_point_left_of_pattern.add(
                    (
                        pyomo_model.point_left_of_pattern[j, i] * M
                        + self.training_set[j, i]
                    )
                    >= pyomo_model.pattern[0, i]
                )
                # if point is not strictly left of pattern, indicator needs to be set to False
                pyomo_model.enforce_point_left_of_pattern.add(
                    self.training_set[j, i] + 1e-3
                    <= (
                        pyomo_model.pattern[0, i]
                        + (1 - pyomo_model.point_left_of_pattern[j, i]) * M
                    )
                )
                # if point is strictly right of pattern, indicator needs to be set to True
                pyomo_model.enforce_point_right_of_pattern.add(
                    self.training_set[j, i]
                    <= (
                        pyomo_model.pattern[1, i]
                        + pyomo_model.point_right_of_pattern[j, i] * M
                    )
                )
                # if point is not strictly right of pattern, indicator needs to be set to False
                pyomo_model.enforce_point_right_of_pattern.add(
                    (
                        (1 - pyomo_model.point_right_of_pattern[j, i]) * M
                        + self.training_set[j, i]
                    )
                    >= (pyomo_model.pattern[1, i]) - 1e-3
                )

            pyomo_model.include_constraint.add(
                # key bit: this constraint enforces that the pyomo_model.included for jth point can be set to 0 only
                # if the point is not contained in the pattern (as witnessed by the fact that the corresponding
                # auxiliary variables are all 0)
                pyomo_model.included[j]
                + sum(
                    pyomo_model.point_right_of_pattern[j, i]
                    + pyomo_model.point_left_of_pattern[j, i]
                    for i in self.drange
                )
                >= 1
            )

        # connect auxiliary variables: interval lengths are differences of pattern points
        # and set bounds of the pattern to be optmized
        pyomo_model.interval_constraint = pyo.ConstraintList()
        pyomo_model.pattern_constraint = pyo.ConstraintList()
        for i in self.drange:
            pyomo_model.pattern_constraint.add(
                pyomo_model.pattern[0, i] >= np.min(self.training_set[:, i]) - 1e-2
            )
            pyomo_model.pattern_constraint.add(
                pyomo_model.pattern[1, i] <= np.max(self.training_set[:, i]) + 1e-2
            )
            pyomo_model.interval_constraint.add(
                pyomo_model.interval_lengths[i]
                == pyomo_model.pattern[1, i] - pyomo_model.pattern[0, i]
            )
        # pyomo_model.interval_lengths.pprint()
        return pyomo_model

    def extract_points_included_in_pattern(self):
        return np.array(
            [
                self.training_set[i]
                for i in self.pyomo_model.included
                if np.round(self.pyomo_model.included[i].value, 1) == 1.0
            ]
        )

    def extract_pattern(self):
        intervals = np.zeros((2, 2), dtype=float)
        for _, j in enumerate(self.pyomo_model.pattern):
            intervals[j] = self.pyomo_model.pattern[j].value
        return intervals.T

    def calculate_largest_bounding_area(self):
        result = np.zeros((self.d, 2), dtype=float)
        for i in self.drange:
            result[i] = np.array(
                [
                    np.min(self.training_set[:, i]) - 1e-2,
                    np.max(self.training_set[:, i] + 1e-2),
                ]
            )
        return result  # np.concatenate(self.largest_bounding_area, tmp)

    def classify(self, point_to_be_classified: np.array, tee):
        """
        This function evaluates one testing point and returns the calculated objective (f_hat)
        """
        # point to be classified is a 1 x d array
        self.add_point_to_model(point_to_be_classified)
        _ = pyo.SolverFactory("mindtpy").solve(
            self.pyomo_model,
            strategy="OA",
            mip_solver="glpk",
            nlp_solver="ipopt",
            tee=tee,
        )
        try:
            res = pyo.value(self.pyomo_model.obj)
        except:
            print(
                "-classify- Something went wrong with the solver: ",
                point_to_be_classified,
            )
            res = None
        finally:
            self.minimized_f_hats = np.round(res, 2) if res is not None else None
            return res

    def add_point_to_model(self, point):
        # point to be classified lies in pattern
        point = point.squeeze()
        # assert point.shape == (2,)
        self.pyomo_model.point_constraint = pyo.ConstraintList()
        for i in self.drange:
            # x[i] <= point[i] <= x[i + d], for all i
            self.pyomo_model.point_constraint.add(
                self.pyomo_model.pattern[0, i] <= point[i]
            )
            self.pyomo_model.point_constraint.add(
                point[i] <= self.pyomo_model.pattern[1, i]
            )
