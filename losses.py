import numpy as np
from abc import ABC


class LossFunction(ABC):
    """
    Class to calculate the defined loss function for a given target vector and a prediction.
    """
    def __init__(self):
        ...

    def __call__(self, output: np.array, target: np.array):
        ...

    def first_derivative(self, output: np.array, target: np.array, **kwargs):
        ...


class Unity(LossFunction):
    def __init__(self):
        super(Unity).__init__()

    def __call__(self, output: np.array, target: np.array):
        return target - output

    def first_derivative(self, output: np.array, target: np.array, **kwargs):
        return np.eye(output.shape[0])


class QuadraticLoss(LossFunction):
    """
    Calculated the quadratic loss or Mean squared error (mse) for regression problems. The target and output are
    expected to have the form 1 x observations
    """
    def __init__(self):
        super(QuadraticLoss).__init__()

    def __call__(self, output: np.array, target: np.array):
        return np.mean(np.linalg.norm(output - target))

    def first_derivative(self, output: np.array, target: np.array, **kwargs):
        return output - target


class CrossEntropy(LossFunction):
    """
    The cross entropy \sum_{classes}y log(y_hat) is calculated for a binary/multiclass classification problem. The
    target vector is assumed to have the form 1 x observations. The target values may not be one-hot encoded, but
    be numerated starting from zero. The output is a matrix of the form classes x observations.
    """
    def __init__(self):
        super(CrossEntropy).__init__()

    def __call__(self, output, target):
        N = target.shape[1]
        p = np.copy(output)
        # binary cross entropy case
        if output.shape[0] == 1:
            loss = target * np.log(np.clip(output, 0, 1) + 1e-8)
            loss += (1 - target) * np.log(1 - np.clip(output, 0, 1) + 1e-8)
            loss = np.mean(-loss)
        # multiclass cross entropy case
        else:
            log_likelihood = -np.log(p[target, range(N)])
            loss = np.sum(log_likelihood) / N
        return loss

    def first_derivative(self, output: np.array, target: np.array, **kwargs):
        a = -(np.divide(target, output+1e-8)) + (np.divide(1-target, 1-output+1e-8))
        return a

    def first_derivative_softmax(self, output, target):
        N = target.shape[1]
        grad = np.copy(output)
        grad[target, range(N)] -= 1
        grad = grad/N
        return grad