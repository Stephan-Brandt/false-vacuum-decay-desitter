from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pathos.multiprocessing as mp
from scipy.optimize import curve_fit

from bounce.bounce_solution import BounceSolution
from gelfand_yaglom.effective_potential import EffectivePotential
from gelfand_yaglom.gelfan_yaglom_model import GelfandYaglomModel
from gelfand_yaglom.vacuum_fluctuation import VacuumFluctuation
from helpers.mathematica_link import MathematicaLink

from model_specific.constant_storage import ConstantStorage
from model_specific.scale_factor import ScaleFactor


class LargeLSolver:
    def __init__(
        self,
        bounce_solution: BounceSolution,
        potential,
        constants,
        min_l=8,
        max_l=100,
        csv_filepath="./solutions/det_values.csv",
        load_dets=False,
        recalculate_vacuum_fluctuation=True,
    ):
        self.bounce_solution = bounce_solution
        self.timeframe = self.bounce_solution.timeframe
        self.potential = potential

        self.min_l = min_l
        self.max_l = max_l

        self.constants = constants
        self.constants_range = [
            ConstantStorage(
                self.constants.beta,
                self.constants.epsilon,
                l,
                self.constants.s_disturbance,
            )
            for l in range(min_l, max_l)
        ]

        self.csv_filepath = csv_filepath
        if load_dets:
            self.dets = np.genfromtxt(self.csv_filepath, delimiter=",")
        else:
            self.dets = None

        self.recalculate_vacuum_fluctuation = recalculate_vacuum_fluctuation

        self.solutions = np.array([])

        self.gy_record = None

        self.cutoff_model = None

        self.starting_time_dependency_record = None

    def multiprocess_det_ratio(self, inplace=True, save=False):
        print("#" * len(self.constants_range))
        pool = mp.Pool(6)
        try:
            gelfand_yaglom_solutions = pool.map(
                self.calculate_det_ratio, reversed(self.constants_range)
            )
            gelfand_yaglom_solutions = np.array(gelfand_yaglom_solutions)
            self.gy_record = gelfand_yaglom_solutions[:, 2]
            self.starting_time_dependency_record = gelfand_yaglom_solutions[:, [0, -1]]
            l_det_log_pairs = gelfand_yaglom_solutions[:, :2]
        except:
            raise ChildProcessError("The multiprocessing crashed")
        finally:
            pool.close()
            pool.join()

        if inplace:
            self.dets = np.hstack(
                [
                    l_det_log_pairs,
                    self.get_final_gy_time().reshape(l_det_log_pairs.shape[0], 1),
                ]
            )
            if save:
                self.save_det(self.dets)
        else:
            if save:
                self.save_det(l_det_log_pairs)
            return l_det_log_pairs

    def multiprocess_vacuum_fluctuation(self):
        print("#" * len(self.constants_range))
        pool = mp.Pool(6)
        try:
            pool.map(self.calculate_vacuum_fluctuations, reversed(self.constants_range))
        except Exception as e:
            raise e
        finally:
            pool.close()
            pool.join()

    def calculate_vacuum_fluctuations(self, constants):
        mathematica_link = MathematicaLink(
            self.bounce_solution,
            self.potential,
            constants,
            recalculate_vacuum_fluctuation=True,
        )

        _ = VacuumFluctuation(
            self.bounce_solution.timeframe,
            self.potential,
            constants,
            mathematica_link,
            recalculate_vacuum_fluctuation=True,
        )

        print("#", end="")

    def calculate_det_ratio(self, constants):
        mathematica_link = MathematicaLink(
            self.bounce_solution,
            self.potential,
            constants,
            recalculate_vacuum_fluctuation=self.recalculate_vacuum_fluctuation,
        )

        try:
            vf = VacuumFluctuation(
                self.bounce_solution.timeframe,
                self.potential,
                constants,
                mathematica_link,
                recalculate_vacuum_fluctuation=self.recalculate_vacuum_fluctuation,
            )
            truncated_timeframe = mathematica_link.dataframe["timeframe"].to_numpy()
            if len(truncated_timeframe):
                bounce_solution_l = self.bounce_solution.interpolate_to_new_timeframe(
                    truncated_timeframe, inplace=False
                )
            else:
                return constants.l, np.nan

            ep = EffectivePotential(
                bounce_solution_l.timeframe,
                bounce_solution_l,
                constants,
                self.potential,
                mathematica_link,
            )
            mathematica_link.delete_singleton()

            gy = GelfandYaglomModel(bounce_solution_l.timeframe, constants, vf, ep)

            starting_time_dependency = []
            for starting_time in reversed(
                range(0, round(len(self.timeframe) / 50), 5,)
            ):
                _ = gy.solve_model(starting_index=starting_time)
                gy.truncate_solution()
                det_log = gy.calculate_det_log()
                starting_time_dependency.append(
                    [gy.timeframe[starting_time], det_log[-1]]
                )

            print("#", end="")
            return (constants.l, det_log[-1], gy, np.array(starting_time_dependency))
        except KeyError as e:
            print(e)
            print("Mathematica Failed at l " + str(constants.l))
        except IndexError as e:
            print("Index error at l " + str(constants.l))
            raise e
        # finally:
        #     mathematica_link.delete_singleton()

    def plot_dets(
        self, include_envelope=True, granularity=1000, show_plot=True, filepath=None
    ):
        if self.dets is None:
            raise ValueError(
                "You need to calculate the dets bafore plotting them. Consider the method 'multiprocess_det_ratio'"
            )
        l_vector = self.dets[:, 0]
        l_range = np.linspace(1, l_vector[-1], granularity)

        det_values = self.dets[:, 1]

        fig = plt.figure(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)
        plt.title(r"Large $\ell$ Behaviour of Ratio Operator")
        plt.ylabel(r"$\ln(T_{\ell}(\tau_{max}))$")
        plt.xlabel(r"$\ell$")

        plt.scatter(l_vector, det_values)
        if include_envelope:
            plt.plot(
                l_range,
                self.calculate_alpha() / (l_range + 1),
                label=r"$\frac{\alpha}{\ell+1}$",
            )
            plt.plot(
                l_range,
                self.calculate_alpha() / (l_range + 1)
                + self.calculate_gamma() / (l_range + 1) ** 3,
                label=r"$\frac{\alpha}{\ell+1}+\frac{\gamma}{\left(\ell+1\right)^{3}}$",
            )
            plt.legend()

        det_range = np.max(det_values) - np.min(det_values)
        plt.ylim(
            [np.min(det_values) - det_range / 2, np.max(det_values) + det_range / 2]
        )

        if filepath is not None:
            plt.savefig(filepath, format="eps")
        if show_plot:
            plt.show()

    def save_plot(self, filepath="./solutions/dets.eps"):
        self.plot_dets(show_plot=False, filepath=filepath)

    def plot_diff(
        self, plot_range=None, save_plot=False, filepath="./solutions/diff.eps"
    ):
        diff = self.calculate_diff()

        fig = plt.figure(figsize=(15, 10))
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)
        plt.title("Difference to WKB")
        plt.ylabel(
            r"$\ln\left(T^{\left(l\right)}\left(\sigma_{\max}\right)\right)-\left(\frac{\alpha}{l+1}+\frac{\gamma}{\left(l+1\right)^{3}}\right)$"
        )
        plt.xlabel(r"$l$")
        if plot_range is not None:
            plt.xlim(plot_range)
        plt.plot(self.dets[:, 0], diff)
        if save_plot:
            plt.savefig(filepath, format="eps")
        plt.show()

    def plot_square_diff(
        self, plot_range=None, save_plot=False, filepath="./solutions/diff_square.eps"
    ):
        diff = self.calculate_square_diff()

        fig = plt.figure(figsize=(15, 10))
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)
        plt.title("Difference to WKB with l-Squared Factor")
        plt.ylabel(
            r"$\left(l+1\right)^{2}\cdot\ln\left(T^{\left(l\right)}\left(\sigma_{\max}\right)\right)-\left(\left(l+1\right)\alpha+\frac{\gamma}{\left(l+1\right)}\right)$"
        )
        plt.xlabel(r"$l$")
        if plot_range is not None:
            plt.xlim(plot_range)
        plt.plot(self.dets[:, 0], diff)
        if save_plot:
            plt.savefig(filepath, format="eps")
        plt.show()

    def plot_partial_sums(
        self,
        plot_range=None,
        save_plot=False,
        include_fit=False,
        filepath="./solutions/cutoff_dependency.eps",
    ):
        cutoff_dependency = self.calculate_cutoff_dependency()

        fig = plt.figure(figsize=(15, 10))
        fig.patch.set_facecolor("white")
        plt.ticklabel_format(useOffset=False)
        plt.title("Cutoff Dependency of Partial Sum")
        plt.ylabel(
            r"$\sum_{\ell}\left[\left(\ell+1\right)^{2}\cdot\ln\left(T^{\left(\ell\right)}\left(\tau_{\max}\right)\right)-\left(\left(\ell+1\right)\alpha+\frac{\gamma}{\left(\ell+1\right)}\right)\right]$"
        )
        plt.xlabel(r"$\ell$-cutoff")
        if plot_range is not None:
            plt.xlim(plot_range)
            plt.plot(
                self.dets[plot_range[0] : plot_range[1], 0],
                cutoff_dependency[plot_range[0] : plot_range[1]],
            )
            if include_fit:
                cutoff_coefficients = self.fit_convergence(stopping_point=40)
                plt.plot(
                    self.dets[plot_range[0] : plot_range[1], 0],
                    self.cutoff_model(
                        self.dets[plot_range[0] : plot_range[1], 0],
                        *cutoff_coefficients[0]
                    ),
                )
        else:
            plt.plot(self.dets[:, 0], cutoff_dependency)
            if include_fit:
                cutoff_coefficients = self.fit_convergence(stopping_point=40)
                plt.plot(
                    self.dets[:, 0],
                    self.cutoff_model(self.dets[:, 0], *cutoff_coefficients[0]),
                )
        if save_plot:
            plt.savefig(filepath, format="eps")
        plt.show()

    def fit_coefficients(self, starting_point=0):
        if starting_point > self.max_l:
            raise ValueError(
                "Startinng point is out of range. The maximum possible value for this instance is "
                + str(self.max_l)
            )
        starting_point = max(self.min_l, starting_point)
        included_dets = self.dets[self.dets[:, 0] > starting_point]

        model = lambda x, alpha, gamma: alpha / (x + 1) + gamma / (x + 1) ** 3
        coefficients = curve_fit(model, included_dets[:, 0], included_dets[:, 1])

        return coefficients

    def calculate_alpha(self):
        scale_factor = ScaleFactor()

        def integrand(t):
            return (
                0.5
                * scale_factor.a(t)
                * (
                    -self.potential.evaluate_second_derivative(
                        self.bounce_solution.bounce
                    )
                    + self.potential.evaluate_second_derivative(
                        self.potential.false_vacuum
                    )
                )
            )

        return np.trapz(
            integrand(self.bounce_solution.timeframe), self.bounce_solution.timeframe
        )

    def calculate_beta(self, time):
        scale_factor = ScaleFactor()

        def calculate_varphi(t):
            return (
                0.25
                * scale_factor.a(t) ** 2
                * (
                    -self.potential.evaluate_second_derivative(
                        self.bounce_solution.bounce
                    )
                    + self.potential.evaluate_second_derivative(
                        self.potential.false_vacuum
                    )
                )
            )

        def interpolate_varphi(t):
            return np.interp(t, self.timeframe, calculate_varphi(self.timeframe)).item()

        interpolation_function = np.vectorize(interpolate_varphi)

        return interpolation_function(time)

    def calculate_gamma(self):
        scale_factor = ScaleFactor()

        def integrand(t):
            return (
                -0.125
                * scale_factor.a(t)
                * (
                    (
                        -self.potential.evaluate_second_derivative(
                            self.bounce_solution.bounce
                        )
                        + self.potential.evaluate_second_derivative(
                            self.potential.false_vacuum
                        )
                    )
                    * (
                        -2
                        - 2 * scale_factor.a_dot(t) ** 2
                        + scale_factor.a(t) ** 2
                        * (
                            -self.potential.evaluate_second_derivative(
                                self.bounce_solution.bounce
                            )
                            - self.potential.evaluate_second_derivative(
                                self.potential.false_vacuum
                            )
                        )
                    )
                )
            )

        return np.trapz(
            integrand(self.bounce_solution.timeframe), self.bounce_solution.timeframe
        )

    def calculate_full_det(self, truncation_point=50):
        truncation_point = min(self.max_l, truncation_point)
        included_dets = self.dets[self.dets[:, 0] < truncation_point]

        diff = (
            (included_dets[:, 0] + 1) ** 2 * included_dets[:, 1]
            - self.calculate_alpha() * (included_dets[:, 0] + 1)
            - self.calculate_beta(self.dets[:, 2])
            - self.calculate_gamma() / (included_dets[:, 0] + 1)
        )

        return diff.sum()

    def calculate_diff(self):
        diff = (
            self.dets[:, 1]
            - self.calculate_alpha() / (self.dets[:, 0] + 1)
            - self.calculate_beta(self.dets[:, 2]) / (self.dets[:, 0] + 1) ** 2
            - self.calculate_gamma() / (self.dets[:, 0] + 1) ** 3
        )

        return diff

    def calculate_square_diff(self):
        diff = (
            (self.dets[:, 0] + 1) ** 2 * self.dets[:, 1]
            - self.calculate_alpha() * (self.dets[:, 0] + 1)
            - self.calculate_beta(self.dets[:, 2])
            - self.calculate_gamma() / (self.dets[:, 0] + 1)
        )

        return 0.5 * diff

    def get_final_gy_time(self):
        get_last_entry = lambda gy: gy.timeframe[-1]
        get_last_entry = np.vectorize(get_last_entry)
        return get_last_entry(self.gy_record)

    def calculate_partial_sums(self, truncation_point=50):
        truncation_point = min(self.max_l, truncation_point)
        included_dets = self.dets[self.dets[:, 0] < truncation_point]

        return ((included_dets[:, 0] + 1) ** 2 * included_dets[:, 1]).sum()

    def calculate_cutoff_dependency(self):
        diffs = self.calculate_square_diff()
        return diffs.cumsum()

    def save_det(self, dets=None):
        if dets is None:
            np.savetxt(
                self.csv_filepath + str(self.constants.beta), self.dets, delimiter=","
            )
        else:
            np.savetxt(self.csv_filepath, dets, delimiter=",")

    def fit_convergence(self, starting_point=0, stopping_point=None):
        if stopping_point is None:
            stopping_point = self.dets[-1, 0]
        self.cutoff_model = (
            lambda x, epsilon, shift, offset: epsilon / ((x + shift) ** 2) + offset
        )
        coefficients = curve_fit(
            self.cutoff_model,
            self.dets[:stopping_point, 0],
            self.calculate_cutoff_dependency()[:stopping_point],
            p0=[1000, 1, -160],
        )

        return coefficients
