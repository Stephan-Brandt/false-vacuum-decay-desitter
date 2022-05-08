from subprocess import call
from typing import Callable

import mpmath as mpm
import numpy as np
import pandas as pd


class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance


# class MathematicaLink(Singleton):
class MathematicaLink:
    def __init__(
        self,
        bounce_solution=None,
        potential=None,
        constants=None,
        csv_filepath="./solutions/mathematica_link",
        recalculate_vacuum_fluctuation=True,
    ):
        self.is_first_instance = self._check_first_instance()

        if self.is_first_instance and (
            bounce_solution is None or potential is None or constants is None
        ):
            raise ValueError(
                "For the first instance of 'MathematicaLink' you need to supply the bounce solution, the potential and "
                "the constants!"
            )
        elif not self.is_first_instance and (
            bounce_solution is not None or potential is not None
        ):
            raise RuntimeWarning(
                "There exists already an instance of 'MathematicaLink'. Your input for 'bounce_solution' and "
                "'potential' will be ignored!"
            )

        self.constants = constants
        self.bounce_solution = bounce_solution
        self.potential = potential
        self.csv_filepath = csv_filepath + "_" + str(self.constants.l) + ".csv"
        if recalculate_vacuum_fluctuation:
            self.export_csv()

        self.dataframe = self.import_csv()

    def import_csv(self) -> pd.DataFrame:
        """
        Import and preprocess data from csv file. It imports all vacuum fluctuation data and cleans up the mathematica
        data types.

        :return: dataframe with the vacuum fluctuation data.
        """

        def replace_in_column(
            old_str: str, new_str: str
        ) -> Callable[[pd.Series], pd.Series]:
            """
            Function for string replacement old with new value.
            :param old_str: pattern to replace
            :param new_str: pattern to substitute
            :return: lambda function to string replace.
            """
            return (
                lambda x: x.str.replace(old_str, new_str, regex=False)
                if x.dtype == object
                else x
            )

        df = pd.read_csv(
            self.csv_filepath, na_values=["NaN", "Indeterminate", "ComplexInfinity"]
        )
        if df.shape[0] != 10000:
            raise RuntimeError("The processed dataframe does not have full size!")

        for old_string, new_string in [(" ", ""), ("I", "j"), ("*^", "e"), ("*", "")]:
            df = df.apply(replace_in_column(old_string, new_string))
        df = df.apply(
            lambda x: x.apply(lambda y: mpm.mpmathify(y)) if x.dtype == object else x
        )
        return df

    def export_csv(self) -> None:
        """
        Export the timeframe, vacuum fluctuation, the normal and logarithmic derivatives to csv.
        """

        data = pd.DataFrame(
            {
                "timeframe": self.bounce_solution.timeframe,
                "bounce": self.bounce_solution.bounce,
                "bounce_first_derivative": self.bounce_solution.first_derivative,
                "bounce_second_derivative": self.bounce_solution.second_derivative,
                "potential_first_derivative": self.potential.evaluate_first_derivative(
                    self.bounce_solution.bounce
                ),
                "potential_second_derivative": self.potential.evaluate_second_derivative(
                    self.bounce_solution.bounce
                ),
            }
        )
        if data.shape[0] != 10000:
            raise RuntimeError("The initial dataframe does not have full size!")

        data.to_csv(self.csv_filepath)

    def _check_first_instance(self):
        try:
            _ = self.is_first_instance
            return False
        except AttributeError:
            return True

    def execute_script(self, script_path, *args, verbose=False):
        shell_command = (
            script_path
            + " '"
            + self.csv_filepath
            + "' "
            + " ".join([str(arg) for arg in [*args]])
        )

        if verbose:
            print(
                "The following shell command is being executed: \n"
                + shell_command
                + "\n"
            )

        status_code = call(shell_command, shell=True,)

        self.dataframe = self.import_csv()
        self.dataframe.dropna(inplace=True)

        return status_code

    def delete_singleton(self):
        self.__delattr__("is_first_instance")
