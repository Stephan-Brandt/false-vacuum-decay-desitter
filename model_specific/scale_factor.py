import numpy as np


class ScaleFactor:
    def __init__(self):
        self.a = lambda chi: np.sin(chi)
        self.a_dot = lambda chi: np.cos(chi)

        self.hubble = lambda chi: self.a_dot(chi) / self.a(chi)
