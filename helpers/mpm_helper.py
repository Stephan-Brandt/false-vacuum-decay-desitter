import numpy as np
from mpmath import mpc, norm


class MPMHelper:
    def __init__(self, values):
        self.values = values

        self.complex_values = self.convert_to_complex()
        if self.values.dtype == complex:
            self.values = self.convert_to_mpc()

        self.zeroth_derivative = None
        self.first_derivative = None
        self.splice_values()

    def truncate_values(self, index):
        self.values[index:, 0] = self.values[index, 0]
        self.values[index:, 1] = 0
        self.splice_values()

    def convert_to_complex(self):
        vectorized_function = np.vectorize(lambda x: complex(x))
        return vectorized_function(self.values)

    def convert_to_mpc(self):
        vectorized_function = np.vectorize(lambda x: mpc(x))
        return vectorized_function(self.values)

    def calculate_norm(self):
        vectorized_function = np.vectorize(lambda x: float(norm(x)))
        return vectorized_function(self.zeroth_derivative)

    def calculate_first_derivative_norm(self):
        vectorized_function = np.vectorize(lambda x: float(norm(x)))
        return vectorized_function(self.first_derivative)

    def splice_values(self):
        self.zeroth_derivative = self.values[:, 0]
        self.first_derivative = self.values[:, 1]
