from losses import *
from activations import *
from backprop_nn import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_data_and_function(X, f, y):
    plt.plot(X, f(X), c='g', lw=2, label="True Function")
    plt.scatter(X, y, c='b', s=5, label="observations")
    plt.title("Non-linear function")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    stdev = 0.39
    X = np.arange(-1, 1, 0.005).reshape(-1, 1)
    f = lambda x: 1/8*(63*x**5 - 70*x**3 + 15*x)
    y = f(X) + np.random.normal(0, stdev, size=X.shape)

    # plot true funciton
    # plot_data_and_function(X, f, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    permutation = X_test.argsort(axis=0).squeeze()
    X_test = X_test[permutation]
    y_test = y_test[permutation]

    # make sure runs have same initialized weights
    rng = np.random.RandomState(123)
    state = rng.__getstate__()

    # define nns
    layers = [1, 20, 10, 1]
    activations = [ReLU(), ReLU(), Linear()]
    epochs_bp = 1
    epochs_ekf = 10
    # Create two identical KNN's that will be trained differently
    nn = NeuralNetwork(layers=layers, activations=activations, loss=QuadraticLoss(), rng=rng)
    loss_sgd, val_loss_sgd = nn.train(
        X_test.T,
        y_test.reshape(1, -1),
        batch_size=1,
        epochs=epochs_bp,
        lr=0.05,
        val=(
            X_test.T,
            f(X_test).reshape(1, -1)
        ),
        momentum=0
    )
    _, y_hat_sgd = nn.feedforward(X_test.T)
    # reset state
    rng.__setstate__(state)

    # train with ekf
    nn = NeuralNetwork(layers=layers, activations=activations, loss=Unity(), rng=rng)
    loss_ekf, val_loss_ekf = nn.train_ekf(
        X_test.T,
        y_test.reshape(1, -1),
        P=100, R=1, Q=1e-2,
        epochs=epochs_ekf,
        val=(
            X_test.T,
            f(X_test).reshape(1, -1)
        ),
        eta=.3

    )
    _, y_hat_ekf = nn.feedforward(X_test.T)

    # Evaluation
    plt.title("Model Fit to Data with $eta=1$")
    plt.scatter(X_test, y_test, c='b', s=5, label="Training data")
    plt.plot(X_test, f(X_test), c='g', lw=2, label="True function")
    plt.plot(X_test, y_hat_ekf[-1].squeeze(), c='k', ls=':', lw=2, label="Kalman {} epochs".format(epochs_ekf))
    plt.legend()
    plt.show()

    plt.plot(val_loss_sgd.keys(), val_loss_sgd.values(), label="SGD")
    plt.plot(val_loss_ekf.keys(), val_loss_ekf.values(), label="EKF")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Test set loss for different epochs")
    plt.legend()
    plt.show()
