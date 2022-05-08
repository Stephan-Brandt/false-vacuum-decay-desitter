import numpy as np

from gelfand_yaglom.vacuum_fluctuation import VacuumFluctuation
from model_specific.constant_storage import ConstantStorage
from model_specific.potential import Potential


class VacuumFluctuationMock(VacuumFluctuation):
    def __init__(
        self, timeframe: np.ndarray, potential: Potential, constants: ConstantStorage
    ):
        self.timeframe = timeframe
        self.potential = potential
        self.constants = constants

        self.logarithmic_derivative = 71 + np.exp(0.0000001 / np.sin(self.timeframe))
