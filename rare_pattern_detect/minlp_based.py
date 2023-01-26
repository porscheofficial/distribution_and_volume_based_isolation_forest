import pyomo.environ as pyo
import numpy as np
from rare_pattern_detect.patterns import PatternSpace  # , min_volume

DEFAULT_SOLVER_SETTINGS = {
    "strategy": "OA",
    "mip_solver": "glpk",
    "nlp_solver": "ipopt",
    # "use_mcpp": "False",
    # "use_fbbt": "False",
    # "threads": 0,
    # "absolute_bound_tolerance": 0.1,
    # "relative_bound_tolerance": 0.01,
    "tee": False,
}


def minlp_has_rare_pattern(
    point_to_be_classified,
    training_data,
    pattern_space: PatternSpace,
    mu,
    testing_data=None,
    solver_settings=None,
):
    if solver_settings is None:
        solver_settings = {}
    min_volume = pattern_space.cutoff
    model = MINLPModel(training_data, testing_data, min_volume)

    # Checking if point is included in the largest bounding area defined by the training set
    if contains(point_to_be_classified, model.largest_bounding_area):
        solution = model.find_min_f_hat(
            point_to_be_classified, solver_settings=solver_settings
        )
        # print(f"evaluating: {point_to_be_classified} -> f_hat:{solution} <?> mu:{mu}")
        # Parse solution output
        if solution is not None:
            # If the minlp pyomo_model was feasible and a solution was found
            # then we return the pyomo_model and the label that contains
            # if the point is anomalous or not (bool)
            res = (model, solution <= mu)
        else:
            # Else if for some reasons the pyomo_model encountered an error
            # while trying to solve the minlp pyomo_model we return (None, None)
            print(
                "Error when classifying a point: ",
                point_to_be_classified,
            )
            res = (model, None)
    else:
        print("point to be classified outside of the limits: anomaly")
        # no need to solve the minlp pyomo_model for this point.
        # Since the point lies outside of the largest point area, then it must be an anomaly (True)
        # This should only happen in case we split the dataset to training and testing set.
        # In the case of unsupervised learning, we consider the whole dataset as a training set
        solution = 0.0
        res = (model, True)

    model.minimized_f_hats = solution
    # model.pyomo_model.included.pprint()
    return res


def contains(point: np.ndarray, largest_bounding_area) -> bool:
    contained = all(
        (largest_bounding_area.T[0, :] <= point.T)
        & (point.T <= largest_bounding_area.T[1, :])
    )
    return contained


class MINLPModel:
    def __init__(self, training_set: np.array, min_volume: float, **kwargs):
        # !! This should never happen -> f_hat is zero -> everything anomaleous"
        # -> A test was added to test for this case
        assert min_volume != 0.0, "min_volume is zero"
        self.kwargs = kwargs
        self.training_set = training_set  # a N x d matrix
        # self.testing_set = testing_set
        self.min_volume = min_volume  # the smallest allowed area
        self.N, self.d = self.training_set.shape
        self.Nrange, self.drange = (range(x) for x in self.training_set.shape)
        self.largest_bounding_area = self.calculate_largest_bounding_area()
        self.pyomo_model = self.create_pyomo_model()
        self.minimized_f_hats = np.array([], dtype=np.float64)
        self.point_to_be_classified = None

    def create_pyomo_model(self):
        def _pattern_area():
            return pyo.prod(pyomo_model.interval_lengths[i] for i in self.drange)

        # define pyomo_model
        pyomo_model = pyo.ConcreteModel()

        # variables

        # x is a 2d vector

        def whole_bounding_area(model, i, j):
            return self.largest_bounding_area.T[i, j]

        def pattern_bounds(model, i, j):
            return tuple(self.largest_bounding_area.T[:, j])

        pyomo_model.pattern = pyo.Var(
            range(2),
            self.drange,
            within=pyo.Reals,
            bounds=pattern_bounds,
            initialize=whole_bounding_area,
        )

        # y is a boolean vector of size N
        pyomo_model.included = pyo.Var(self.Nrange, within=pyo.Binary, initialize=0)

        # Constraint to bound the number of points included in the pattern
        # to at most 10% of the total number of training points
        if "bound_included" in self.kwargs:
            bound_included = round(self.kwargs["bound_included"] * self.N)
            pyomo_model.constraint_included_point_in_pattern = pyo.Constraint(
                expr=sum(pyomo_model.included[i] for i in self.Nrange) <= bound_included
            )

        # auxiliary variables
        def initializing_area_func(model, i):
            return float(
                self.largest_bounding_area[i, 1] - self.largest_bounding_area[i, 0]
            )

        pyomo_model.interval_lengths = pyo.Var(
            self.drange,
            within=pyo.NonNegativeReals,
            initialize=initializing_area_func,
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

        # pattern area needs to exceed min_volume
        pyomo_model.area_constraint = pyo.Constraint(
            expr=_pattern_area() >= self.min_volume
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
        """
        This returns the bounding area, with a small addition in both sides, and returns it in the shape (d,2):
        """
        result = np.zeros((self.d, 2), dtype=float)
        for i in self.drange:
            result[i] = np.array(
                [
                    np.min(self.training_set[:, i]) - 1e-2,
                    np.max(self.training_set[:, i]) + 1e-2,
                ]
            )
        return result

    def find_min_f_hat(self, point_to_be_classified: np.array, solver_settings=None):
        """
        This function evaluates one testing point and returns the calculated objective (f_hat)
        """
        if solver_settings is None:
            solver_settings = {}

        # point to be classified is a 1 x d array
        self.add_point_to_model(point_to_be_classified)

        # initialise the initial_pattern if given
        if "initial_pattern" in self.kwargs:

            if self.kwargs["initial_pattern"] == "maximal":
                # auxiliary variables
                def initializing_area_func(model, i):
                    return float(
                        self.largest_bounding_area[i, 1]
                        - self.largest_bounding_area[i, 0]
                    )

                def whole_bounding_area(model, i, j):
                    return self.largest_bounding_area.T[i, j]

                self.pyomo_model.interval_lengths.initialize = initializing_area_func
                self.pyomo_model.included.initialise = 1
                self.pyomo_model.pattern.initialise = whole_bounding_area

            elif self.kwargs["initial_pattern"] == "minimal":
                nth_root = self.min_volume ** (1 / self.d)

                def calc_min_side_lengths(model, i, j):
                    return point_to_be_classified[j] - ((-1) ** i) * nth_root

                self.pyomo_model.included.initialise = 0
                self.pyomo_model.interval_lengths.initialize = nth_root
                self.pyomo_model.pattern.initialise = calc_min_side_lengths

        # Use the nth root of the min_volume over two in all nth dimensions to initialize the pattern
        # self.pyomo_model.pattern[0, i] = self.point_to_be_classified[i] - (
        #     self.nth_root / 2
        # )
        #
        # self.pyomo_model.pattern[1, i] = self.point_to_be_classified[i] + (
        #     self.nth_root / 2
        # )

        for k, v in DEFAULT_SOLVER_SETTINGS.items():
            if k not in solver_settings:
                solver_settings[k] = v

        # Old slow solver
        # _ = pyo.SolverFactory("mindtpy").solve(
        #     self.pyomo_model,
        #     strategy="OA",
        #     mip_solver="glpk",
        #     nlp_solver="ipopt",
        #     tee=tee,
        # )

        # # New faster solver
        # _ = pyo.SolverFactory("mindtpy").solve(
        #     self.pyomo_model,
        #     strategy="OA",
        #     mip_solver="gurobi_persistent",
        #     nlp_solver="appsi_ipopt",
        #     use_mcpp="True",
        #     use_fbbt="True",
        #     threads=4,
        #     # absolute_bound_tolerance=0.1,
        #     # relative_bound_tolerance=0.01,
        #     tee=tee,
        # )
        # New faster solver
        _ = pyo.SolverFactory("mindtpy").solve(self.pyomo_model, **solver_settings)
        try:
            res = pyo.value(self.pyomo_model.obj)
        except:
            print(
                "-classify- Something went wrong with the solver: ",
                point_to_be_classified,
            )
            res = None
        finally:
            # self.minimized_f_hats = np.round(res, 2) if res is not None else None
            return res

    def add_point_to_model(self, point):
        # point to be classified lies in pattern
        self.point_to_be_classified = point.squeeze()

        # if np.any(self.testing_set):
        #     # print("extracting index from testing set")
        #     lst = self.testing_set.tolist()
        # else:
        #     lst = self.training_set.tolist()
        #
        # index = lst.index(self.point_to_be_classified.tolist())

        self.pyomo_model.point_constraint = pyo.ConstraintList()
        # self.pyomo_model.point_constraint.add(self.pyomo_model.included[index] == 1)
        for i in self.drange:
            # x[i] <= point[i] <= x[i + d], for all i
            self.pyomo_model.point_constraint.add(
                self.pyomo_model.pattern[0, i] <= self.point_to_be_classified[i]
            )
            self.pyomo_model.point_constraint.add(
                self.point_to_be_classified[i] <= self.pyomo_model.pattern[1, i]
            )
