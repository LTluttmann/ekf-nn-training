from abc import ABC
import numpy as np


class ActivationFunction(ABC):
    def __init__(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        ...

    def first_derivative(self, *args, **kwargs):
        ...


class Linear(ActivationFunction):
    def __init__(self, c=None):
        super(ActivationFunction).__init__()
        self.c = c if c else 1

    def __call__(self, x):
        return self.c*x

    def first_derivative(self, x):
        return self.c


class ReLU(ActivationFunction):
    def __init__(self):
        super(ActivationFunction).__init__()

    def __call__(self, x):
        return np.where(x <= 0, 0, x)

    def first_derivative(self, x):
        return np.where(x <= 0, 0, 1)


class Sigmoid(ActivationFunction):
    def __init__(self):
        super(ActivationFunction).__init__()

    def __call__(self, x: np.array):
        return np.exp(x) / (1 + np.exp(x))

    def first_derivative(self, x):
        a = self(x)
        return a * (1 - a)


class Softmax(ActivationFunction):
    def __init__(self):
        super(ActivationFunction).__init__()

    def __call__(self, x: np.array):
        a = np.exp(x)
        return a / (np.sum(a, axis=0, keepdims=True))

    def first_derivative(self, x):
        raise NotImplementedError("Use Softmax activation only together with Cross-Entropy Loss")