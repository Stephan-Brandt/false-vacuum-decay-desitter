from typing import List

import gif
import numpy as np
import pandas as pd
from jax import numpy as jnp
from matplotlib import pyplot as plt

from model_specific.constant_storage import ConstantStorage
from model_specific.potential import Potential


class BounceSolution:
    """
    Class for analyzing the bounce solution for the differential equation.
    """

    def __init__(
        self,
        timeframe: np.ndarray,
        solution: np.ndarray,
        potential: Potential,
        plateau_precision: float = 0.0005,
    ):
        """
        :param timeframe: Time interval over which the solution is valid
        :param solution: solution found by the model class
        :param potential: the potential the field moves in
        :param plateau_precision: the maximum (absolute) slope at which a point is classified as part of a plateau
        """
        self.timeframe = timeframe

        self.bounce = solution[:, 0]
        self.first_derivative = solution[:, 1]
        self.second_derivative = np.gradient(
            self.first_derivative, self.timeframe[1] - self.timeframe[0]
        )

        self.potential = potential

        self.plateau_precision = plateau_precision
        self.truncation_time = self.timeframe[-1]

    def integrate_bounce(self) -> float:
        """
        Calculate the area under the bounce curve.
        :return: the integrated bounce
        """
        return np.trapz(self.bounce, self.timeframe)

    def find_convergence_plateau(self) -> np.ndarray:
        """
        Locate plateau at which the solution approximates the false vacuum.
        :return: subset of the bounce solution that lies on the plateau at the false vacuum
        """
        is_false_vacuum_plateau = self._find_plateau_location()
        plateau = self.bounce[is_false_vacuum_plateau]

        return plateau

    def truncate_solution(self, solution_precision) -> None:
        """
        Truncate the solution before divergence. The solution is truncated at the false vacuum turning point for
        overshooting solutions and at the false vacuum minimum for undershooting solutions.
        """
        is_false_vacuum_plateau = self._find_plateau_location()

        false_vacuum_location = np.argmin(
            np.abs(solution_precision[is_false_vacuum_plateau])
        ).item()
        index = self._find_index_from_plateau_index(
            false_vacuum_location, is_false_vacuum_plateau
        )

        if np.any(index):
            truncation_location = np.min(index)

            self.truncation_time = self.timeframe[truncation_location]
            print("last time:", self.timeframe[-1])
            self.bounce[truncation_location:] = self.bounce[truncation_location]
            self.first_derivative[truncation_location:] = 0
            self.second_derivative[truncation_location:] = 0
        else:
            raise RuntimeWarning(
                "No truncation point found. Solution was not truncated"
            )

    def interpolate_to_new_timeframe(
        self, new_timeframe: np.ndarray, inplace: bool = True
    ):
        """
        The solution is linearly interpolated to any new timeframe between the minimum and maximum time step.
        :param new_timeframe: numpy array with the new time values
        """
        if np.any(np.diff(new_timeframe) < 0):
            raise RuntimeError(
                "The new timeframe must be strictly monotonically increasing!"
            )
        # if (
        #     new_timeframe[0] < self.timeframe[0]
        #     or new_timeframe[-1] > self.timeframe[-1]
        # ):
        #     raise RuntimeError(
        #         "The new timeframe must be a subset of the original timeframe!"
        #     )

        if inplace:
            self.bounce = np.interp(new_timeframe, self.timeframe, self.bounce)
            self.first_derivative = np.interp(
                new_timeframe, self.timeframe, self.first_derivative
            )
            self.second_derivative = np.interp(
                new_timeframe, self.timeframe, self.second_derivative
            )
            self.timeframe = new_timeframe
        else:
            bounce_soluion = self

            bounce_soluion.bounce = np.interp(
                new_timeframe, self.timeframe, self.bounce
            )
            bounce_soluion.first_derivative = np.interp(
                new_timeframe, self.timeframe, self.first_derivative
            )
            bounce_soluion.second_derivative = np.interp(
                new_timeframe, self.timeframe, self.second_derivative
            )
            bounce_soluion.timeframe = new_timeframe
            return bounce_soluion

    @gif.frame
    def plot_solution(
        self, overlay_derivative: bool = False, show_plot=True, filepath=None
    ) -> None:
        """
        Plot the solution found
        :param overlay_derivative: Boolean whether to plot the solution's first derivative as well
        :param show_plot: boolean whether to display plot
        :param filepath: place to save the figure in
        """
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)

        plt.plot(self.timeframe, self.bounce, "b", label="Bounce Solution")
        if overlay_derivative:
            plt.plot(
                self.timeframe, self.first_derivative, "r", label="first derivative   "
            )
            plt.axhline(self.plateau_precision)
            plt.axhline(-self.plateau_precision)

        plt.ylim(
            [
                self.potential.false_vacuum - self.potential.vacuum_distance / 2,
                self.potential.true_vacuum + self.potential.vacuum_distance / 2,
            ]
        )

        plt.legend(loc="best")
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$\phi_{bounce}$")

        if filepath is not None:
            plt.savefig(filepath, format="eps")
        if show_plot:
            plt.show()

    def save_plot(self, filepath="./solutions/bounce_solution", **kwargs):
        self.plot_solution(show_plot=False, filepath=filepath)

    def save_to_csv(
        self,
        constants: ConstantStorage,
        potential: Potential,
        filepath: str = "./solutions/bounce_solution",
    ) -> None:
        """
        Save the bounce values and the first two derivatives to csv.
        :param constants: model constants
        :param potential: the potential of the problem
        :param filepath: the filepath to save the csv to
        """
        dataframe = pd.DataFrame(
            data={
                "timeframe": self.timeframe,
                "bounce": self.bounce,
                "first_derivative": self.first_derivative,
                "second_derivative": self.second_derivative,
            }
        )

        dataframe.to_csv(
            filepath
            + "_beta_"
            + str(constants.beta)
            + "_b_"
            + str(potential.b)
            + ".csv"
        )

    def _find_plateau_location(self) -> List[bool]:
        """
        Find the indices at which the field approximates the false_vacuum.
        :return: a boolean list with True in all plateau points
        """
        is_plateau = np.abs(self.first_derivative) < self.plateau_precision
        is_false_vacuum = (
            np.abs(self.bounce - self.potential.false_vacuum)
            < np.abs(self.potential.false_vacuum - self.potential.true_vacuum) / 10
        )
        return is_plateau & is_false_vacuum

    @staticmethod
    def _find_index_from_plateau_index(
        plateau_index: int, is_false_vacuum_plateau: List[bool]
    ) -> int:
        """
        Function to find the corresponding index of the bounce solution when a index of the plateau values only is
        given.

        :param plateau_index: indices in the array of false vacuum plateau values to truncate at
        :param is_false_vacuum_plateau: list with True if in false vacuum plateau, False else
        :return: indices of the bounce solution corresponding to the plateau indices
        """

        counter = sum(is_false_vacuum_plateau[:plateau_index])
        index = plateau_index

        while counter < plateau_index:
            counter += is_false_vacuum_plateau[index]
            index += 1

        return index

    def _find_turning_points(self) -> List[bool]:
        """
        Find the turning points of the bounce solutions via the roots of the second derivative.
        :return: Boolean list with true at turning points
        """
        curvature_sign = np.sign(self.second_derivative)
        turnng_points = ((np.roll(curvature_sign, 1) - curvature_sign) != 0).astype(
            bool
        )
        turnng_points[0] = 0

        return turnng_points

    def cast_to_jnp(self):
        self.timeframe = jnp.array(self.timeframe)
        self.bounce = jnp.array(self.bounce)
        self.first_derivative = jnp.array(self.first_derivative)
        self.second_derivative = jnp.array(self.second_derivative)
