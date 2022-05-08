import gif
import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar
from scipy.signal import find_peaks

from bounce.bounce_model import BounceModel
from bounce.bounce_solution import BounceSolution
from model_specific.potential import Potential


class ModelShooter:
    """
    Shooting procedure for finding the bounce solution.
    """

    def __init__(
        self,
        model: BounceModel,
        potential: Potential,
        max_steps: int = 1000,
        required_precision: float = 1e-6,
    ):
        """
        :param model: the model describing the differential equation of the bounce solution
        :param potential: the potential the field is in
        :param max_steps: the maximum number of iteration steps
        :param required_precision: the precision required between the plateau and the false vacuum
        """
        self.model = model
        self.potential = potential

        self.max_steps = max_steps
        self.current_step = None

        self.required_precision = required_precision

        self.precision = self.potential.true_vacuum - self.potential.false_vacuum
        self.precision_log = np.array([])

        self.initial_positions_log = np.array([])
        self.initial_position = None

        self.upper_bound = self.potential.true_vacuum
        self.upper_bound_log = np.array([])

        self.lower_bound = self.potential.false_vacuum
        self.lower_bound_log = np.array([])

        self.minimum_log = np.array([])

        self.solution = None
        self.gif_data = []

    # This is very much meant literally!
    def shoot_model(
        self, starting_position: float = None, save_gif=False
    ) -> BounceSolution:
        """
        Vary the initial position until the boundary condition phi(pi)=false_vacuum
        :param starting_position: the initial condition for phi
        :param save_gif: boolean whether to save the solutions as a gif
        :return: the final solution found
        """
        self.gif_plots = []
        if starting_position is None:
            self.initial_position = (self.upper_bound + self.lower_bound) / 2
        else:
            self.initial_position = starting_position

        self.current_step = 0

        plots = []
        with ProgressBar(max_value=self.max_steps) as bar:
            while self.current_step < self.max_steps:
                self.solution = self.model.solve_model([self.initial_position, 0])
                self._save_iteration_markers()
                try:
                    self._update_initial_position()
                    if save_gif:
                        plot = self.solution.plot_solution(show_plot=False)
                        plots.append(plot)

                        self.gif_data.append(self.solution)
                except SystemExit:
                    print(
                        "\nPrecision achieved with final precision "
                        + str(self.precision)
                        + "\n"
                    )
                    break
                self.current_step += 1
                bar.update(self.current_step)

            else:
                self.solution = self.model.solve_model([self.initial_position, 0])

                print(
                    "\nIteration limit reached!\n"
                    + "Final precision: "
                    + str(self.precision)
                    + "\n"
                )
                bar.update(self.max_steps)

        if save_gif:
            self.create_gif(plots)
        solution_precision = self.model.check_solution()
        self.solution.truncate_solution(solution_precision)
        print(self.solution.timeframe[-1])
        return self.solution

    def create_gif(self, plots):
        gif.save(
            plots,
            "./solutions/bounce_solution.gif",
            duration=10,
            unit="s",
            between="startend",
        )

    def plot_iteration_markers(self) -> None:
        """
        Plot the current initial position, as well as the upper and lower bound dependent on the iteration steps.
        """
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)

        plt.plot(self.upper_bound_log, "r", label="Upper Bound")
        plt.plot(self.lower_bound_log, "g", label="Lower Bound")
        plt.plot(self.initial_positions_log, "b", label="Initial Position")

        plt.xlabel("Iteration Step")
        plt.ylabel("Marker")
        plt.legend(loc="best")

        plt.show()

    def plot_precision(self) -> None:
        """
        Plot the precision at which the solution fulfills the boundary condition phi(pi)=false_vacuum
        """
        fig = plt.figure()
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)

        plt.plot(self.precision_log, "r")

        plt.ylim(
            [
                self.potential.false_vacuum - self.potential.vacuum_distance / 2,
                self.potential.true_vacuum + self.potential.vacuum_distance / 2,
            ]
        )

        plt.xlabel("Iteration Step")
        plt.ylabel("Precision")

        plt.show()

    def _update_initial_position(self) -> None:
        """
        Adjust initial position in a such that it moves towards the Coleman DeLuccia Bounce.
        """
        peak_positions = find_peaks(-self.solution.bounce)[0]

        if self.precision is not None and self.precision < self.required_precision:
            raise SystemExit("\nSolution found.\n")
        elif len(peak_positions):
            first_minimum = self.solution.bounce[peak_positions[0]]
        else:
            first_minimum = np.min(self.solution.bounce)

        if (
            np.all(self.solution.bounce >= self.potential.true_vacuum)
            or first_minimum < self.potential.false_vacuum
        ):
            self._shift_to_lower_bound()
        elif self.potential.false_vacuum < first_minimum < self.potential.true_vacuum:
            self._shift_to_upper_bound()

        else:
            raise RuntimeError("\nUnable to update initial position!\n")

    def _shift_to_lower_bound(self) -> None:
        """
        Shifting initial position "down the hill" so there is less overshoot
        """
        self.upper_bound = self.initial_position
        self.initial_position = (self.upper_bound + self.lower_bound) / 2
        self._save_iteration_markers()

    def _shift_to_upper_bound(self) -> None:
        """
        Shifting initial position "up the hill" so that the field "goes further up the hill"
        """
        self.lower_bound = self.initial_position
        self.initial_position = (self.upper_bound + self.lower_bound) / 2
        self._save_iteration_markers()

    def _save_iteration_markers(self) -> None:
        """
        Save the current initial position, as well as the upper and lower bound of the initial position to their
        respective logs
        """
        self.initial_positions_log = np.append(
            self.initial_positions_log, self.initial_position
        )
        self.lower_bound_log = np.append(self.lower_bound_log, self.lower_bound)
        self.upper_bound_log = np.append(self.upper_bound_log, self.upper_bound)

        self._update_precision()
        self.precision_log = np.append(self.precision_log, self.precision)

    def _update_precision(self) -> None:
        """
        Calculate and save the precision at which the last solution fulfills the boundary condition phi(pi)=false_vacuum
        """
        plateau = self.solution.find_convergence_plateau()

        if len(plateau):
            self.precision = np.max(np.abs(plateau - self.potential.false_vacuum))
        else:
            self.precision = None
        self.precision_log = np.append(self.precision_log, self.precision)
