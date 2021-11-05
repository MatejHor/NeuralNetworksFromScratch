from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        print("Epochs: {}, LearningRate: {}, Layer: {}".format(epochs, l_rate, sizes))
        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def relu(self, x, derivative=False):
        if derivative:
            func = lambda x: 0.0 if x <= 0.0 else 1.0
        else:
            func = lambda x: 0.0 if x <= 0.0 else x
        return np.vectorize(func)(x)

    def initialization(self):
        params = {}
        layers = self.sizes
        for index in range(1, len(layers)):
            params['W' + str(index)] = np.random.randn(layers[index], layers[index - 1]) * np.sqrt(1. / layers[index])
            params['b' + str(index)] = np.zeros((layers[index]))
        return params

    def forward_pass(self, x_train):
        params = self.params
        len_layers = len(self.sizes)

        # input layer activations becomes sample
        params['A0'] = x_train

        for index in range(1, len_layers):
            # input layer to hidden layer 1
            params['Z' + str(index)] = np.dot(params["W" + str(index)], params['A' + str(index - 1)]) + params['b' + str(index)]
            params['A' + str(index)] = self.relu(params['Z' + str(index)])

        return params['A' + str(len_layers - 1)]

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_w = {}
        len_layers = len(self.sizes)

        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z' + str(len_layers - 1)], derivative=True)
        for index in reversed(range(1, len_layers)):
            # Calculate W3 update
            change_w['W' + str(index)] = np.outer(error, params['A' + str(index - 1)]) 
            change_w['b' + str(index)] = error

            # Calculate W2 update
            if index != 1:
                error = np.dot(params['W' + str(index)].T, error) * self.relu(params['Z' + str(index - 1)], derivative=True)
        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''
        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        al = []

        predictions = []
        loss = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))      

            log = np.log(output)
            loss.append(y * np.where(log == -np.Inf, 0, log))

            # Output from forward
            al.append(pred)
        
        return np.mean(predictions), (-np.sum(loss)), al

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            
            accuracy, loss, al = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%, Loss: {3:.4f}'.format(
                iteration+1, time.time() - start_time, accuracy * 100, loss
            ))
            
# import nn_python_impl_on_numma_mnist as nn
# # # x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# # # x_ = x.to_numpy()

file = open("./data/fashion_mnist_train_vectors.csv")
x_train = np.loadtxt(file, delimiter=',')
file = open("./data/fashion_mnist_train_labels.csv")
y_train = np.loadtxt(file, delimiter=',')

file = open("./data/fashion_mnist_test_vectors.csv")
x_test = np.loadtxt(file, delimiter=',')
file = open("./data/fashion_mnist_test_labels.csv")
y_test = np.loadtxt(file, delimiter=',')

x_train = (x_train/255).astype('float64')
y_train = to_categorical(y_train).astype('float64')

x_test = (x_test/255).astype('float64')
y_test = to_categorical(y_test).astype('float64')

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.164, random_state=42)

print("x_train.shape ", x_train.shape)
print("y_train.shape ", y_train.shape)
print("x_val.shape ", x_val.shape)
print("y_val.shape ", y_val.shape)
print("x_test.shape ", x_test.shape)
print("y_test.shape ", y_test.shape)

dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10], l_rate=0.01, epochs=10)
dnn.train(x_train, y_train, x_val, y_val)

acc, loss, al = dnn.compute_accuracy(x_test, y_test)
print('--\n', al, '--\n')
# print('y_test:', y_test)

print("Test Accuracy: {:.2f}%, Loss: {:.4f}".format(acc * 100, loss))