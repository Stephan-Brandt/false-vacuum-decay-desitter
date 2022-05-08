class ConstantStorage:
    def __init__(
        self, beta: float, epsilon: float, l: int, s_disturbance: int
    ) -> None:
        self.beta = beta
        self.epsilon = epsilon
        self.l = l
        self.s_disturbance = s_disturbance
