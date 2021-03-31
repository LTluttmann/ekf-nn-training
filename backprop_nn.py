import numpy as np
from activations import Softmax, ActivationFunction
from losses import CrossEntropy, Unity, QuadraticLoss, LossFunction
from typing import List, Tuple
from copy import deepcopy


class NeuralNetwork(object):

    def __init__(
            self,
            layers: list,
            activations: List[ActivationFunction],
            loss: LossFunction,
            rng: np.random._generator.Generator = None
    ):
        assert (len(layers) == len(activations) + 1)
        self.layers = layers
        self.activations = activations
        self.loss = loss
        self.weights = []
        self.biases = []
        self.rng = np.random.RandomState(123) if not rng else rng
        for l in range(len(layers) - 1):
            # xavier initialization of weights
            self.weights.append(np.sqrt(2 / layers[l]) * self.rng.randn(layers[l + 1], layers[l]))
            self.biases.append(np.zeros((layers[l + 1], 1)))

    def feedforward(self, x: np.ndarray) -> Tuple[list, list]:
        """
        Function to feed forward a signal x through the network
        :param x: signal / feature matrix
        :return: list of weighted inputs and activations for each layer
        """
        # return the feedforward value for x
        a = np.copy(x)
        z_s = []
        a_s = [a]
        for l in range(len(self.weights)):
            z_s.append(self.weights[l].dot(a) + self.biases[l])
            a = self.activations[l](z_s[-1])
            a_s.append(a)
        return z_s, a_s

    def backpropagation(self, y: np.ndarray, z_s: list, a_s: list) -> Tuple[list, list]:
        """
        Function that implements the backpropagation algorithm
        :param y: 1xobservation numpy array containing the labels
        :param z_s: list containing array of weighted inputs calculated at each neuron for each layer
        :param a_s: list containing array of activations calculated at each neuron for each layer
        :return: list containing gradients of cost function w.r.t weights and biases for each layer
        """
        # delta = dC/dZ  known as error for each layer
        deltas = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        # insert the last layer error
        # first derivative of cross entropy has an easy form if last layers activation is softmax. Call it here
        if isinstance(self.activations[-1], Softmax) and isinstance(self.loss, CrossEntropy):
            deltas[-1] = np.atleast_2d(
                self.loss.first_derivative_softmax(a_s[-1], y)
            )
        # otherwise, the first derivative is the hadamard product of gradient of loss and gradient of output
        else:
            deltas[-1] = np.atleast_2d(
                self.loss.first_derivative(a_s[-1], y) * (self.activations[-1].first_derivative(z_s[-1]))
            )
        # perform backpropagation
        for l in reversed(range(len(deltas) - 1)):
            deltas[l] = self.weights[l + 1].T.dot(deltas[l + 1]) * (
                self.activations[l].first_derivative(z_s[l])
            )
        batch_size = y.shape[1]  # observations in columns, features/classes in rows
        # calculate the derivatives of the loss w.r.t weights using the deltas (error terms)
        db = [d.dot(np.ones((batch_size, 1))) / float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T) / float(batch_size) for i, d in enumerate(deltas)]
        return dw, db

    def train(
            self, x_train, y_train, batch_size=10, epochs=100, lr=0.01, val: tuple = None, momentum=0
    ) -> Tuple[dict, dict]:
        """
        Function that implements stochastic gradient descent with standard backpropagation for training the
        neural network. If a validation set is provided, the best parameters are saved and overwrite the
        current weights and biases at the end of the training
        :param x_train: input features of training data. Required shape is umpy 2D array with D rows and N columns
        :param y_train: labels of training data. Required shape is numpy 2D array with 1 row and N columns
        :param batch_size: size of the mini batch
        :param epochs: number of training iterations
        :param lr: learning rate alpha
        :param val: validation set (features, labels)
        :param momentum: momentum parameter, if momentum should be used
        :return: dictionary of train set and validation set loss for the different epochs
        """
        # update weights and biases based on the output
        losses = {}
        val_losses = {}
        # define the function to be used to evaluate the loss
        eval_func = lambda x, y: self.loss(self.feedforward(x)[1][-1], y)
        # save the best parameters in terms of validation set loss
        best_weights = self.weights.copy()
        best_biases = self.biases.copy()

        # initialize velocity vectors for sgd with momentum
        weight_velocity = [
            np.zeros_like(weights) for weights in self.weights
        ]
        bias_velocity = [
            np.zeros_like(biases) for biases in self.biases
        ]

        for e in range(epochs):
            x, y = self.shuffle_training_data(x_train, y_train)
            i = 0
            while i < y.shape[1]:
                x_batch = x[:, i:i + batch_size]
                y_batch = y[:, i:i + batch_size]
                i = i + batch_size
                z_s, a_s = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                if momentum:
                    weight_velocity = [
                        momentum * velocity - lr * dweight for velocity, dweight in zip(weight_velocity, dw)
                    ]
                    bias_velocity = [
                        momentum * velocity - lr * dbias for velocity, dbias in zip(bias_velocity, db)
                    ]
                    self.weights = [
                        w + velocity for w, velocity in zip(self.weights, weight_velocity)
                    ]
                    self.biases = [
                        w + velocity for w, velocity in zip(self.biases, bias_velocity)
                    ]
                else:
                    self.weights = [
                        w - lr * weight_gradient for w, weight_gradient in zip(self.weights, dw)
                    ]
                    self.biases = [
                        w - lr * bias_gradient for w, bias_gradient in zip(self.biases, db)
                    ]
            losses[e] = eval_func(x, y)
            if val:
                val_losses[e] = eval_func(*val)
                if val_losses[e] <= min(val_losses.values()):
                    # save the best parameters in terms of validation set loss
                    best_weights = deepcopy(self.weights)
                    best_biases = deepcopy(self.biases)
            else:
                val_losses[e] = np.nan
            print("Training loss = {}. Validation loss = {}".format(losses[e], val_losses[e]))
        self.weights = best_weights
        self.biases = best_biases
        return losses, val_losses

    def train_ekf(
            self, x, y, P=None, Q=None, R=None, epochs=10, val: tuple = None, eta=0.3
    ) -> Tuple[dict, dict]:
        """
        this function executes the EKF-algorithm iteratively for training the neural network. It iterates
        through the specified number of training iterations and within each training iteration through every
        training pattern provided in arrays x and y. If a validation set is provided, the best parameters are
        saved and overwrite the current weights and biases at the end of the training.
        :param x: input features of training data. Required shape is numpy 2D array with D rows and N columns
        :param y: labels of training data. Required shape is numpy 2D array with 1 row and N columns
        :param P: square matrix of size equal to the number of parameters or scalar value
        :param R: square matrix of size equal to the number of output neurons or scalar value
        :param Q: square matrix of size equal to the number of parameters, scalar value or None
        :param epochs: number of training iterations
        :param val: tuple containing a validation set: (features, labels)
        :param eta: forgetting factor for updating Q and R
        :return: dictionary of train set and validation set loss for the different epochs
        """
        losses = {}
        val_losses = {}
        self._init_kalman(P, R, Q)
        best_weights = self.weights.copy()
        best_biases = self.biases.copy()

        # Require that loss function used in backpropagation is the identity, i.e. make Jacobian of
        # output w.r.t. weights
        if not isinstance(self.loss, Unity):
            eval_func = lambda x, y: self.loss(self.feedforward(x)[1][-1], y)
            self.loss = Unity()
        else:
            eval_func = lambda x, y: QuadraticLoss()(self.feedforward(x)[1][-1], y)
        # iterate over epochs and training instances
        for e in range(epochs):
            x_shuffled, y_shuffled = self.shuffle_training_data(x, y)
            # Train
            for i in range(y.shape[1]):
                x_batch = x_shuffled[:, [i]]
                y_batch = y_shuffled[:, [i]]
                # Forward propagation
                z_s, a_s = self.feedforward(x_batch)
                # Do the learning
                self._ekf(x_batch, y_batch, z_s, a_s, eta)
            losses[e] = eval_func(x, y)
            if val:
                val_losses[e] = eval_func(*val)
                if val_losses[e] <= min(val_losses.values()):
                    best_weights = deepcopy(self.weights)
                    best_biases = deepcopy(self.biases)
            else:
                val_losses[e] = np.nan
            print("Training loss = {}. Validation loss = {}".format(losses[e], val_losses[e]))
            print("New R: {}, or learning rate: {}".format(self.R, 1/self.R))
        self.weights = best_weights
        self.biases = best_biases

        return losses, val_losses

    def shuffle_training_data(self, x_train, y_train) -> Tuple[np.ndarray, np.ndarray]:
        """
        In order to randomize the training process, the training data is randomly permuted.
        :param x_train: design matrix of training instances
        :param y_train: labels of training instances
        :param rng: Random number generator
        :return: shuffled dataset
        """
        x_train = np.copy(x_train)
        y_train = np.copy(y_train)
        permutation = self.rng.permutation(y_train.shape[1])
        y_train = np.take(y_train, permutation, axis=1)
        x_train = np.take(x_train, permutation, axis=1)
        return x_train, y_train

    def _init_kalman(self, P, R, Q):
        """
        Initialize P, Q and R matrices. One can only pass scalar values for the respective matrices to the ekf train
        method of the neural network. In this case, the matrices P, Q and R are constructed by multiplying the
        scalar with the identity matrix.
        """
        num_params = sum(map(np.size, self.weights)) + sum(map(np.size, self.biases))
        if np.isscalar(P):
            self.P = P*np.eye(num_params)
        else:
            assert P.shape == (num_params, num_params)
            self.P = P

        if Q is None:
            self.Q = np.zeros((num_params, num_params))
        elif np.isscalar(Q):
            self.Q = Q*np.eye(num_params)
        else:
            assert Q.shape == (num_params, num_params)
            self.Q = Q

        if np.isscalar(R):
            self.R = R*np.eye(self.layers[-1])

    def _ekf(self, x, y, z_s, a_s, eta):
        """
        This function implements the Extended Kalman Filter Recursion that determines the weight
        update for the neural network.
        """
        # Compute NN jacobian
        # perform backpropagation
        Hw, Hb = self.backpropagation(y, z_s, a_s)
        # NN Jacobian H is a N x W matrix, with N the number of instances and W the number of
        # parameters. Stack the weights and biases horizontally in this step
        Hw = np.hstack([dweight.flatten() for dweight in Hw])
        Hb = np.hstack([dbias.flatten() for dbias in Hb])
        H = np.hstack((Hw, Hb)).reshape(1, -1)

        # compute Kalman gain
        A = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(A))

        # compute the weight delta
        innovation = self.loss(a_s[-1], y)
        dtheta = K.dot(innovation)

        # split dthetha into weight and bias gradients
        dw_flat = dtheta[:Hw.size]
        db_flat = dtheta[Hw.size:]

        # bring the gradients in the same shape as the weight and bias matrices
        idx_weight, idx_bias = 0, 0

        # update the weights and biases
        for i, (layer_weights, layer_biases) in enumerate(zip(self.weights, self.biases)):
            self.weights[i] += dw_flat[idx_weight:idx_weight+layer_weights.size].reshape(
                layer_weights.shape
            )
            self.biases[i] += db_flat[idx_bias:idx_bias+layer_biases.size].reshape(
                layer_biases.shape
            )
            idx_weight += layer_weights.size
            idx_bias += layer_biases.size

        if eta < 1:
            # update measurement noise
            a_posteriori_y = self.feedforward(x)[-1]
            residual = self.loss(a_posteriori_y[-1], y)
            self.R = eta * self.R + (1-eta) * (np.dot(residual.T, residual) + H.dot(self.P).dot(H.T))
            # update process noise
            self.Q = eta * self.Q + (1-eta) * np.dot(K, np.dot(innovation.T, innovation)).dot(K.T)
        # update error covariance
        self.P = self.P - np.dot(K, H.dot(self.P)) + self.Q
