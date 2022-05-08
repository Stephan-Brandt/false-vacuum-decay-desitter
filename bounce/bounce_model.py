import warnings
from typing import Union

import numpy as np
from scipy.integrate import odeint

from bounce.bounce_solution import BounceSolution
from model_specific.potential import Potential


class BounceModel:
    """
    Implementing the differential equation for Bounce solutions.
    """

    def __init__(
        self,
        tw: float,
        potential: Potential,
        tol: float = 1e-5,
        t_min: float = 0,
        t_max: float = np.pi,
        t_granularity: int = 10000,
    ):
        """
        :param tw: pre-factor for the friction term
        :param potential: potential the field moves in
        :param tol: the differential equation is solved on the interval [tol, pi-tol]
        :param t_min: minimum time the differential equation is integrated from
        :param t_max: maximum time the differential equation is integrated to
        :param t_granularity: number of data points of the solution
        """
        self.tw = tw
        self.potential = potential
        self.timeframe = np.linspace(t_min + tol, t_max - tol, t_granularity)
        self.model = self._build_model()
        self.solution = None

    def _build_model(self):
        """
        Build the differential equation for scipy.integrate.odeint to solve.
        :return: Function expressing the differential equation
        """

        def model(y, t):
            """
            Implementation of the differential equation
            :param y: last evaluation
            :param t: timestep
            :return: the next evaluation value
            """
            theta, omega = y

            dydt = [
                omega,
                -self._friction(t) * omega
                - self.potential.evaluate_first_derivative(theta),
            ]
            return dydt

        return model

    def _friction(self, t: Union[float, np.ndarray]) -> float:
        """
        Calculate the friction term
        :param t: the time at which to evaluate the friction
        :return: the friction value at the given time
        """
        return self.tw * 3 / np.tan(t)

    def solve_model(self, initial_conditions: [float, float]) -> BounceSolution:
        """
        Find solution for the differential equation
        :param initial_conditions: List of [phi(t=0), phi_dot(t=0)]
        :return: Solution with phi in first column and phi_dot in second column
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sol = odeint(self.model, initial_conditions, self.timeframe, rtol=1e-10, atol=1e-10)

        self.solution = BounceSolution(self.timeframe, sol, self.potential)
        return BounceSolution(self.timeframe, sol, self.potential)

    def check_solution(self):
        zeroth_derivative = self.solution.bounce
        first_derivative = self.solution.first_derivative
        second_derivative = self.solution.second_derivative
        return (
            second_derivative
            + 3 / np.tan(self.timeframe) * first_derivative
            + self.potential.evaluate_first_derivative(zeroth_derivative)
        )
