from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.np_utils import to_categorical
import numpy as np

# import
# mnist = fetch_openml('mnist_784')
# X, y = mnist["data"], mnist["target"]

file = open("./data/fashion_mnist_train_vectors.csv")
X = np.loadtxt(file, delimiter=',')
file = open("./data/fashion_mnist_train_labels.csv")
y = np.loadtxt(file, delimiter=',')
X = (X/255).astype('float64')

file = open("./data/fashion_mnist_test_vectors.csv")
x_test = np.loadtxt(file, delimiter=',')
file = open("./data/fashion_mnist_test_labels.csv")
y_test = np.loadtxt(file, delimiter=',')
x_test = (x_test/255).astype('float64')

# one-hot encode labels
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

digits = 10
examples = y_test.shape[0]
y_test = y_test.reshape(1, examples)
Y_test = np.eye(digits)[y_test.astype('int32')]
Y_test = Y_test.T.reshape(digits, examples)

# split, reshape, shuffle
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X.T, x_test.T
Y_train, Y_test = Y_new, Y_test
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("Y_train: ", Y_train.shape)
print("Y_test: ", Y_test.shape)

def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum
    return L

def relu(x, derivative=False):
        if derivative:
            func = lambda x: 0.0 if x <= 0.0 else 1.0
        else:
            func = lambda x: 0.0 if x <= 0.0 else x
        return np.vectorize(func)(x)

def feed_forward(X, params):

    cache = {}

    cache["Z1"] = np.dot(params["W1"], X) + params["b1"]
    # cache["A1"] = relu(cache["Z1"])
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.dot(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = softmax(cache['Z2'])

    return cache

def back_propagate(X, Y, params, cache):

    dZ2 = cache["A2"] - Y
    dW2 = (1./m_batch) * np.dot(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    # dZ1 = dA1 * relu(cache["Z1"], derivative=True)
    dW1 = (1./m_batch) * np.dot(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

np.random.seed(138)

# hyperparameters
n_x = X_train.shape[0]
n_h = 256
learning_rate = 4
beta = .9
batch_size = 256
batches = -(-m // batch_size)

# initialization
params = { "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
           "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
           "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
           "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h) }

V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)

    # train
for i in range(20):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):
        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache)

        V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])

        params["W1"] = params["W1"] - learning_rate * V_dW1
        params["b1"] = params["b1"] - learning_rate * V_db1
        params["W2"] = params["W2"] - learning_rate * V_dW2
        params["b2"] = params["b2"] - learning_rate * V_db2

    cache = feed_forward(X_train, params)
    train_cost = compute_loss(Y_train, cache["A2"])
    cache = feed_forward(X_test, params)
    test_cost = compute_loss(Y_test, cache["A2"])
    print("Epoch {}: training cost = {}, test cost = {}".format(i+1 ,train_cost, test_cost))

print("Done.")

cache = feed_forward(X_test, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)

print(classification_report(predictions, labels))

with open('prediction.csv', 'w') as f:
    np.savetxt(f, predictions)