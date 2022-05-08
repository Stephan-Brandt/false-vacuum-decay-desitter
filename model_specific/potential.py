import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from model_specific.constant_storage import ConstantStorage


class Potential:
    """
    Phi^4 Potential for the false vacuum decay
    """

    def __init__(
        self,
        b: float,
        constants: ConstantStorage,
        phi_min: float = -2,
        phi_max: float = 2,
    ) -> None:
        """
        :param b: Coefficient of the phi^3 term.
        :param constants: all the problem wide constants.
        :param phi_min: Lower x-bound for display in plot.
        :param phi_max: Upper x-bound for display in plot.
        """
        self.b = b
        self.constants = constants
        self.phi_min = phi_min
        self.phi_max = phi_max

        self.potential_function = lambda phi: 1 / (
            self.constants.epsilon ** 2
        ) + self.constants.beta * (
            1 / 2 * phi ** 2 + self.b / 3 * phi ** 3 - 1 / 4 * phi ** 4
        )

        self.first_derivative_function = lambda phi: self.constants.beta * (
            phi + self.b * phi ** 2 - phi ** 3
        )

        self.second_derivative_function = lambda phi: self.constants.beta * (
            1 + 2 * self.b * phi - 3 * phi ** 2
        )

        self.false_vacuum, self.true_vacuum = self._find_vacua()

        self.vacuum_distance = self.true_vacuum - self.false_vacuum

    def evaluate_potential(self, phi_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate the symbolic potential function on numerical values.
        :param phi_grid: numpy array to evaluate the function on.
        :return: numpy array with the potential values.
        """
        return self.potential_function(phi_grid)

    def evaluate_first_derivative(self, phi_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate the symbolic first derivative potential function on numerical values.
        :param phi_grid: numpy array to evaluate the function on.
        :return: numpy array with the derivative values.
        """
        return self.first_derivative_function(phi_grid)

    def evaluate_second_derivative(self, phi_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate the symbolic first derivative potential function on numerical values.
        :param phi_grid: numpy array to evaluate the function on.
        :return: numpy array with the derivative values.
        """
        return self.second_derivative_function(phi_grid)

    def plot_potential(
        self, granularity: int = 100000, show_plot=True, filepath=None
    ) -> None:
        """
        Plot the potential.
        :param granularity: Number of datapoints to display
        :param show_plot: boolean whether to display the figure
        :param filepath: place to save the figure in
        """
        phi_grid = np.linspace(self.phi_min, self.phi_max, granularity)
        plt.plot(phi_grid, self.evaluate_potential(phi_grid))

        plt.xlim([self.phi_min, self.phi_max])
        plt.xlabel("phi")
        plt.ylabel("V")
        plt.grid()

        if filepath is not None:
            plt.savefig(filepath)
        if show_plot:
            plt.show()

    def save_plot(self, filepath="./solutions/potential", **kwargs):
        self.plot_potential(filepath=filepath, **kwargs)

    def _find_vacua(self) -> Tuple[float, float]:
        """
        Find the false and true vacuum the bounce interpolates inbetween.
        :return: tuple with (false_vacuum, true_vacuum)
        """
        return (
            1 / 2 * (self.b - math.sqrt(self.b ** 2 + 4)),
            1 / 2 * (self.b + math.sqrt(self.b ** 2 + 4)),
        )
