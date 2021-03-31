from losses import *
from activations import *
from backprop_nn import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    np.random.seed(1234)
    X, Y = np.mgrid[-1:1.1:0.1, -1:1.1:0.1]
    xy = np.vstack((X.flatten(), Y.flatten())).T
    yy = np.sign(np.product(xy, axis=1))
    yy = np.where(yy < 0, 0, 1)

    x_train, x_test, y_train, y_test = train_test_split(xy, yy)
    # make sure runs have same initialized weights
    rng = np.random.RandomState(123)
    state = rng.__getstate__()

    # Create two identical KNN's that will be trained differently
    nn = NeuralNetwork(layers=[2, 20, 10, 1], activations=[ReLU(), ReLU(), Sigmoid()], loss=QuadraticLoss(), rng=rng)
    train_loss, val_loss = nn.train(
        x_train.T, y_train.reshape(1, -1), batch_size=4, epochs=50, lr=0.05, val=(x_test.T, y_test.reshape(1, -1))
    )

    plt.plot(train_loss.keys(), train_loss.values(), label="train")
    plt.plot(val_loss.keys(), val_loss.values(), label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss for Backpropagation")
    plt.legend()
    plt.show()

    # create a mesh to plot in
    h = .02
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    x1, x2 = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = np.round(nn.feedforward(np.c_[x1.ravel(), x2.ravel()].T)[-1][-1], 0)

    # Put the result into a color plot
    Z = Z.reshape(x1.shape)
    ax.contourf(x1, x2, Z, cmap=plt.cm.Pastel1)
    # ax.axis('off')

    # Plot also the training points
    ax.scatter(xy[:, 0], xy[:, 1], c=yy.reshape(1,-1), cmap=plt.cm.tab10)

    ax.set_title('Backpropagation')
    plt.show()
    del fig, ax

    # reset state
    rng.__setstate__(state)

    # train with ekf
    nn = NeuralNetwork(layers=[2, 20, 10, 1], activations=[ReLU(), ReLU(), Linear()], loss=Unity(), rng=rng)
    train_loss, val_loss = nn.train_ekf(
        x_train.T,
        y_train.reshape(1, -1),
        P=100, R=10, Q=1e-2,
        epochs=10,
        val=(x_test.T, y_test.reshape(1, -1)),
        eta=.3
    )

    plt.plot(train_loss.keys(), train_loss.values(), label="train")
    plt.plot(val_loss.keys(), val_loss.values(), label="validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Loss for EKF-Algorithm")
    plt.legend()
    plt.show()

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = np.round(nn.feedforward(np.c_[x1.ravel(), x2.ravel()].T)[-1][-1], 0)

    # Put the result into a color plot
    Z = Z.reshape(x1.shape)
    ax.contourf(x1, x2, Z, cmap=plt.cm.Pastel1)
    #ax.axis('off')

    # Plot also the training points
    ax.scatter(xy[:, 0], xy[:, 1], c=yy.reshape(1,-1), cmap=plt.cm.tab10)

    ax.set_title('EKF Algorithm')
    plt.show()