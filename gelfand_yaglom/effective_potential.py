import numpy as np
from matplotlib import pyplot as plt

from bounce.bounce_solution import BounceSolution
from model_specific.constant_storage import ConstantStorage
from model_specific.potential import Potential
from model_specific.scale_factor import ScaleFactor


class EffectivePotential:
    """
    Helper quantity defined by Dunne in arXiv:hep-th/0605176 in equation 3.3.
    """

    def __init__(
        self,
        timeframe: np.ndarray,
        bounce_solution: BounceSolution,
        constants: ConstantStorage,
        potential: Potential,
        script_path="./mathematica_backend/EffectivePotentialScript",
    ):
        """
        :param timeframe: the timeframe over which to apply the Gelfand-Yaglom method
        :param constants: the problem specific constants
        :param bounce_solution: the bounce solution obtained by the shooting method
        :param potential: the potential of the problem
        """
        self.timeframe = timeframe
        self.bounce_solution = bounce_solution
        self.constants = constants
        self.potential = potential
        self.scale_factor = ScaleFactor()

        self.script_path = script_path

        self.effective_potential = (
            -self.potential.evaluate_second_derivative(self.bounce_solution.bounce)
            + constants.l * (constants.l + 2) / self.scale_factor.a(self.timeframe) ** 2
        )
        self.false_effective_potential = (
            -self.potential.evaluate_second_derivative(self.potential.false_vacuum)
            + constants.l * (constants.l + 2) / self.scale_factor.a(self.timeframe) ** 2
        )

        self.difference_effective_potential = -self.potential.evaluate_second_derivative(
            self.bounce_solution.bounce
        ) + self.potential.evaluate_second_derivative(
            self.potential.false_vacuum
        )

    def interpolate_effective_potential(self):
        return lambda t: np.interp(t, self.timeframe, self.effective_potential).item()

    def interpolate_false_effective_potential(self):
        return lambda t: np.interp(
            t, self.timeframe, self.false_effective_potential
        ).item()

    def interpolate_difference_effective_potential(self):
        return lambda t: np.interp(
            t, self.timeframe, self.difference_effective_potential
        ).item()

    def plot_effective_potential(
        self,
        plot_range_restrictor: int = None,
        show_plot: bool = True,
        filepath: str = None,
    ) -> None:
        """
        Plot the effective potential evaluated on the bounce solution over the timeframe t.
        :param plot_range_restrictor: number of indices to discard from the left and right edge
        :param show_plot: display the plot or not
        :param filepath: path to save the plot to
        """
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)

        if plot_range_restrictor is None:
            plt.plot(self.timeframe, self.effective_potential)
        else:
            plt.plot(
                self.timeframe[plot_range_restrictor:-plot_range_restrictor],
                self.effective_potential[plot_range_restrictor:-plot_range_restrictor],
            )
        plt.title("Fluctuation Potential")
        plt.xlabel(r"$\sigma$")
        plt.ylabel("U")

        if filepath is not None:
            plt.savefig(filepath)
        if show_plot:
            plt.show()

    def plot_false_effective_potential(
        self,
        plot_range_restrictor: int = None,
        show_plot: bool = True,
        filepath: str = None,
    ) -> None:
        """
        Plot the effective potential evaluated on the bounce solution over the timeframe t.
        :param plot_range_restrictor: number of indices to discard from the left and right edge
        :param show_plot: display the plot or not
        :param filepath: path to save the plot to
        """
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)

        if plot_range_restrictor is None:
            plt.plot(self.timeframe, self.false_effective_potential)
        else:
            plt.plot(
                self.timeframe[plot_range_restrictor:-plot_range_restrictor],
                self.false_effective_potential[
                    plot_range_restrictor:-plot_range_restrictor
                ],
            )

        plt.title("Free Fluctuation Potential")
        plt.xlabel(r"$\sigma$")
        plt.ylabel("U")

        if filepath is not None:
            plt.savefig(filepath)
        if show_plot:
            plt.show()

    def save_plot(
        self,
        plot_range_restrictor: int = None,
        filepath="./solutions/effective_potential",
    ):
        self.plot_effective_potential(
            plot_range_restrictor=plot_range_restrictor,
            show_plot=False,
            filepath=filepath,
        )
