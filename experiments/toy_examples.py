from losses import *
from activations import *
from backprop_nn import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler


def breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    num_features = X.shape[1]

    nn_ekf = NeuralNetwork([num_features, 20, 10, 1], activations=[ReLU(), ReLU(), Sigmoid()], loss=Unity())
    nn_ekf.train_ekf(
        X_train.T,
        y_train.reshape(1, -1),
        P=100, R=10, Q=1e-2,
        eta=1, epochs=10,
        val=(X_test.T, y_test.reshape(1, -1))
    )
    y_hat = np.round(nn_ekf.feedforward(X_test.T)[1][-1], 0)
    print("EKF Accuracy: {}".format(np.sum(y_hat == y_test) / len(y_test)))

    nn = NeuralNetwork([num_features, 20, 10, 1], activations=[ReLU(), ReLU(), Sigmoid()], loss=CrossEntropy())
    nn.train(
        X_train.T,
        y_train.reshape(1, -1),
        epochs=10, batch_size=1, lr=.05,
        val=(X_test.T, y_test.reshape(1, -1))
    )
    y_hat = np.round(nn.feedforward(X_test.T)[1][-1], 0)
    print("Accuracy: {}".format(np.sum(y_hat == y_test) / len(y_test)))


def diabetes():
    X, y = load_diabetes(return_X_y=True)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = sc.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    num_features = X.shape[1]

    nn_ekf = NeuralNetwork([num_features, 20, 10, 1], activations=[ReLU(), ReLU(), Linear()], loss=Unity())
    nn_ekf.train_ekf(
        X_train.T,
        y_train.reshape(1, -1),
        P=100, R=100, Q=1e-2,
        eta=.3, epochs=10,
        val=(X_test.T, y_test.reshape(1, -1))
    )
    y_hat = nn_ekf.feedforward(X_test.T)[1][-1]
    print("EKF loss: {}".format(QuadraticLoss()(y_hat, y_test.reshape(1, -1))))

    nn = NeuralNetwork([num_features, 20, 10, 1], activations=[ReLU(), ReLU(), Linear()], loss=QuadraticLoss())
    nn.train(
        X_train.T,
        y_train.reshape(1, -1),
        epochs=10, batch_size=1, lr=.05,
        val=(X_test.T, y_test.reshape(1, -1))
    )
    y_hat = nn.feedforward(X_test.T)[1][-1]
    print("Backprop loss: {}".format(QuadraticLoss()(y_hat, y_test.reshape(1, -1))))


if __name__ == '__main__':
    diabetes()
