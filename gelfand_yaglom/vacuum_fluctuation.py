from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import mpmath as mpm

from helpers.mathematica_link import MathematicaLink
from model_specific.constant_storage import ConstantStorage
from model_specific.potential import Potential


class VacuumFluctuation:
    """
    Solution for the free fluctuation operator (in the false vacuum) as specified by Dunne in arXiv:hep-th/0605176 in
    equation 4.8.
    """

    def __init__(
        self,
        timeframe: np.ndarray,
        potential: Potential,
        constants: ConstantStorage,
        mathematica_link: MathematicaLink,
        script_path: str = "./mathematica_backend/VacuumFluctuationScript",
        recalculate_vacuum_fluctuation=True
    ) -> None:
        """
        :param timeframe: the timeframe to evaluate the solution over (and later solve Gelfand-Yaglom with)
        :param potential: the potential of the problem
        :param l: the integer label for Gelfand-Yaglom where l(l+2) is the eigenvalue of the S^3 Laplacian
        :param use_mathematica: calculate the vacuum solution and corresponding derivatives with mathematica (=True) or
                mpmath (=False).
        """
        self.timeframe = timeframe
        self.potential = potential
        self.constants = constants

        self.mathematica_link = mathematica_link
        self.script_path = script_path

        if recalculate_vacuum_fluctuation:
            self.mathematica_link.execute_script(
                self.script_path,
                -self.potential.evaluate_second_derivative(self.potential.false_vacuum),
                self.constants.l,
            )
            sleep(1)

        self.mathematica_link.dataframe.dropna(inplace=True)
        self.timeframe = self.mathematica_link.dataframe["timeframe"]

        self.logarithmic_derivative = self.mathematica_link.dataframe[
            "vacuum_fluctuation_logarithmic_derivative_mathematica"
        ]

    def interpolate_logarithmic_derivative(self):
        return lambda t: np.interp(
            t, self.timeframe, self.logarithmic_derivative
        ).item()

    def interpolate_logarithmic_derivative_norm(self):
        norm = np.vectorize(lambda x: np.float(mpm.norm(x)))
        return lambda t: np.interp(
            t, self.timeframe, norm(self.logarithmic_derivative)
        ).item()

    def plot_logarithmic_derivative(self) -> None:
        """
        Plot norm of the vacuum fluctuation solution over the timeframe.
        """
        fig = plt.figure(figsize=(15, 15))
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)
        plt.plot(self.timeframe, np.abs(self.logarithmic_derivative))
        plt.yscale("log")

        plt.show()
