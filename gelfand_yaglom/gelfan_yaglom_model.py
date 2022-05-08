import warnings
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from gelfand_yaglom.effective_potential import EffectivePotential
from gelfand_yaglom.vacuum_fluctuation import VacuumFluctuation
from helpers.mpm_helper import MPMHelper
from model_specific.constant_storage import ConstantStorage
from model_specific.scale_factor import ScaleFactor


class GelfandYaglomModel:
    """
    Class for solving the Gelfand-Yaglom differential equation as described by Dunne in arXiv:hep-th/0605176 in
    equation 4.10.
    """

    def __init__(
        self,
        timeframe: np.ndarray,
        constants: ConstantStorage,
        vacuum_fluctuation: VacuumFluctuation = None,
        effective_potential: EffectivePotential = None,
        plateau_percentile: float = 50,
        gradient_percentile: float = 5,
    ):
        """
        :param vacuum_fluctuation: fluctuation operator for the free/ false vacuum case
        :param effective_potential: effective potential for the specified solution
        :param constants: constants for the problem
        """
        self.constants = constants

        self.timeframe = timeframe

        self.vacuum_fluctuation = vacuum_fluctuation
        self.effective_potential = effective_potential

        self.plateau_percentile = plateau_percentile
        self.gradient_percentile = gradient_percentile

        self.model = self._build_model()
        self.counter = 0

        self.solution = None
        self.det_log = None

        self.truncation_time = None

    def calculate_det_log(self, solution: MPMHelper = None) -> np.ndarray:
        if solution is None:
            solution = self.solution
        norm = solution.calculate_norm()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            log_norm = np.log(norm)

        return log_norm

    def solve_model(
        self, initial_conditions: [float, float] = [1, 0], starting_index=0
    ) -> MPMHelper:
        """
        Find solution for the differential equation.
        :param initial_conditions: Arraylike of [Phi(t=0), Phi_dot(t=0)]
        :return: Solution with phi in first column and phi_dot in second column
        """
        sol = odeint(self.model, initial_conditions, self.timeframe[starting_index:],)

        self.solution = MPMHelper(sol)
        self.det_log = self.calculate_det_log()

        return MPMHelper(sol)

    def plot_det(self, show_plot: bool = True, filepath: str = None):
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)

        plt.title("Ratio Operator for l=" + str(self.constants.l))
        plt.ylabel(r"$\ln (T_{l}(\sigma))$")
        plt.xlabel(r"$\sigma$")

        plt.plot(self.timeframe, self.det_log)

        if filepath is not None:
            plt.savefig(filepath)
        if show_plot:
            plt.show()

    def save_plot(self, filepath="./solutions/det_solution"):
        self.plot_det(show_plot=False, filepath=filepath)

    def _build_model(self) -> Callable[[List[float], float], List[float]]:
        """
        Build the differential equation for a complex adaption of scipy.integrate.odeint to solve.
        :return: Function expressing the differential equation
        """

        def model(y: List[float], chi: float) -> List[float]:
            """
                Implementation of the differential equation
                :param y: last evaluation
                :param chi: timestep
                :return: the next evaluation value
                """
            T, T_dot = y
            dydx = [
                T_dot,
                -(
                    2
                    * self.vacuum_fluctuation.interpolate_logarithmic_derivative()(chi)
                    + 3 * ScaleFactor().hubble(chi)
                )
                * T_dot
                + (
                    self.effective_potential.interpolate_difference_effective_potential()(
                        chi
                    )
                )
                * T,
            ]
            return dydx

        return model

    def truncate_solution(self) -> None:
        """
        Truncate the solution before divergence. The solution is truncated at the false vacuum turning point for
        overshooting solutions and at the false vacuum minimum for undershooting solutions.
        """
        precision = self.check_solution()
        last_tenth = int(np.ceil(len(self.solution.zeroth_derivative) / 10))
        index = int(np.argmin(np.abs(precision[-last_tenth:])))
        absolute_position = int(len(self.solution.zeroth_derivative) - last_tenth + index)
        self.solution.truncate_values(absolute_position - 1)

    def _find_plateau_location(self) -> np.ndarray:
        """
        Find the indices at which the field approximates the false_vacuum.
        :return: a boolean list with True in all plateau points
        """
        first_derivative = np.gradient(self.det_log)
        is_plateau = np.abs(first_derivative) < np.percentile(
            np.abs(first_derivative), self.gradient_percentile
        )

        is_lower_plateau = self.det_log < np.percentile(
            self.det_log, self.plateau_percentile
        )

        return is_plateau & is_lower_plateau

    @staticmethod
    def _find_index_from_plateau_index(
        plateau_index: int, is_lower_plateau: np.ndarray
    ) -> int:
        """
        Function to find the corresponding index of the bounce solution when a index of the plateau values only is
        given.

        :param plateau_index: indices in the array of false vacuum plateau values to truncate at
        :param is_lower_plateau: list with True if in false vacuum plateau, False else
        :return: indices of the bounce solution corresponding to the plateau indices
        """

        counter = sum(is_lower_plateau[:plateau_index])
        index = plateau_index

        while counter < plateau_index:
            counter += is_lower_plateau[index]
            index += 1

        return index

    def _find_turning_points(self) -> np.ndarray:
        """
        Find the turning points of the bounce solutions via the roots of the second derivative.
        :return: Boolean list with true at turning points
        """
        second_derivative = np.gradient(np.gradient(self.det_log))
        curvature_sign = np.sign(second_derivative)
        turning_points = ((np.roll(curvature_sign, 1) - curvature_sign) != 0).astype(
            bool
        )
        turning_points[0] = 0

        return np.array(turning_points)

    def check_solution(self):
        length = min(
            len(self.vacuum_fluctuation.logarithmic_derivative),
            len(self.solution.zeroth_derivative),
        )
        zeroth_derivative = self.solution.zeroth_derivative[:length]
        first_derivative = self.solution.first_derivative[:length]
        second_derivative = np.gradient(
            first_derivative, self.timeframe[1] - self.timeframe[0]
        )[:length]

        return (
            -second_derivative[:length]
            - (
                2 * self.vacuum_fluctuation.logarithmic_derivative[:length]
                + 3 * ScaleFactor().hubble(self.timeframe)[:length]
            )
            * first_derivative[:length]
            + self.effective_potential.difference_effective_potential[:length]
            * zeroth_derivative[:length]
        )
