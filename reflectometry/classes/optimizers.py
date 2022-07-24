import numpy as np

class SGD(object):
    def __init__(self, variables, step_size):
        self.variables = variables
        self.step_size = step_size
        self.delta = np.zeros_like(variables.shape)

    def step(self, grad):

        self.variables -= self.step_size * grad
        self.variables[self.variables>1] = 1
        self.variables[self.variables<0] = 0

    def __repr__(self):
        return "SGD"

class MomentumSGD(object):
    def __init__(self, variables, step_size, alpha):
        self.variables = variables
        self.step_size = step_size
        self.alpha = alpha
        self.delta = np.zeros_like(variables.shape)

    def step(self, grad):
        self.delta = self.alpha * self.delta - (1 - self.alpha) * self.step_size * grad

        self.variables += self.delta
        self.variables[self.variables>1] = 1
        self.variables[self.variables<0] = 0

    def reset(self):
        self.delta *= 0

    def __repr__(self):
        return f"MSGD: alpha={self.alpha:10.0e}"