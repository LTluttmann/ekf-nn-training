from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from losses import *
from activations import *
from backprop_nn import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

if __name__ == "__main__":
    # stdev = 0.01
    # X = np.arange(-10, 10, 0.2)
    # y = np.exp(-X**2) + 0.5*np.exp(-(X-3)**2) + np.random.normal(0, stdev, len(X))
    # sc = StandardScaler()
    # sc.fit(X.reshape(-1, 1))
    # X = sc.transform(X.reshape(-1, 1))
    np.random.seed(1234)
    stdev = 0.1
    X = np.arange(-1, 1, 0.005).reshape(-1, 1)
    f = lambda x: 1/8*(63*x**5 - 70*x**3 + 15*x)
    y = f(X) + np.random.normal(0, stdev, size=X.shape)
    nn = NeuralNetwork(layers=[1, 20, 10, 1], activations=[ReLU(), ReLU(), Linear()], loss=QuadraticLoss())
    nn.train(X.reshape(1, -1), y.reshape(1, -1), batch_size=32, epochs=500, lr=0.2, momentum=.5)
    _, y_hat_sgd = nn.feedforward(X.reshape(1, -1))

    initializer = tf.keras.initializers.HeNormal()
    model = Sequential([
        Dense(20, activation='relu', input_shape=X.shape, kernel_initializer=initializer, bias_initializer="zeros"),
        Dense(10, activation='relu', kernel_initializer=initializer, bias_initializer="zeros"),
        Dense(1, activation='linear', kernel_initializer=initializer, bias_initializer="zeros"),
    ])
    model.compile(
        optimizer=SGD(learning_rate=0.1, momentum=.4),
        loss='mse',
    )
    model.fit(
        X,
        y,
        epochs=500,
        batch_size=32
    )
    keras_y_hat = model.predict(X)

    # Evaluation
    plt.suptitle("Data Fit", fontsize=22)
    plt.scatter(X, y, c='b', s=5)
    plt.plot(X, y_hat_sgd[-1].squeeze(), c='g', lw=3, label="Backprop")
    plt.plot(X, keras_y_hat.squeeze(), c='k', ls=':', lw=2, label="Keras")
    plt.grid(True)
    plt.legend(fontsize=22)
    plt.show()