from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001, optimalizer='GD'):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.optimalizer = optimalizer

        self.params = self.initialization()
        
        if optimalizer == 'adam':
            self.v, self.s = self.initialize_adam(self.params)
        elif optimalizer == 'momentum':
            self.v = self.initialize_velocity(self.params)

        print("Epochs: {}, LearningRate: {}, Optimalizer {}, Layer: {}".format(epochs, l_rate, optimalizer, sizes))

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

        error = 2 * (output - y_train) / output.shape[0] #* self.softmax(params['Z' + str(len_layers - 1)], derivative=True)
        for index in reversed(range(1, len_layers)):
            change_w['W' + str(index)] = np.outer(error, params['A' + str(index - 1)]) 
            change_w['b' + str(index)] = error

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

    def initialize_velocity(self, parameters):
        """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        
        Returns:
        v -- python dictionary containing the current velocity.
                        v['dW' + str(l)] = velocity of dWl
                        v['db' + str(l)] = velocity of dbl
        """
        
        L = len(self.sizes) - 1 # number of layers in the neural networks
        v = {}
        
        # Initialize velocity
        for l in range(L):
            ### START CODE HERE ### (approx. 2 lines)
            v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
            v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
            ### END CODE HERE ###
            
        return v
    
    def update_parameters_with_momentum(self, grads, beta=0.9, learning_rate=0.01):
        """
        Update parameters using Momentum
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- python dictionary containing the current velocity:
                        v['dW' + str(l)] = ...
                        v['db' + str(l)] = ...
        beta -- the momentum hyperparameter, scalar
        learning_rate -- the learning rate, scalar
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- python dictionary containing your updated velocities
        """

        L = len(self.sizes) - 1 # number of layers in the neural networks
        
        # Momentum update for each parameter
        for l in range(L):
            
            ### START CODE HERE ### (approx. 4 lines)
            # compute velocities
            self.v["dW" + str(l+1)] = beta * self.v["dW" + str(l+1)] + (1-beta) * grads['W' + str(l+1)]
            self.v["db" + str(l+1)] = beta * self.v["db" + str(l+1)] + (1-beta) * grads['b' + str(l+1)]
            # update parameters
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - learning_rate * self.v["dW" + str(l+1)]
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - learning_rate * self.v["db" + str(l+1)]
            ### END CODE HERE ###         

    def initialize_adam(self, parameters) :
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL" 
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters["W" + str(l)] = Wl
                        parameters["b" + str(l)] = bl
        
        Returns: 
        v -- python dictionary that will contain the exponentially weighted average of the gradient.
                        v["dW" + str(l)] = ...
                        v["db" + str(l)] = ...
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                        s["dW" + str(l)] = ...
                        s["db" + str(l)] = ...

        """
        
        L = len(self.sizes) - 1 # number of layers in the neural networks
        v = {}
        s = {}
        
        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
        ### START CODE HERE ### (approx. 4 lines)
            v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
            v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
            s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
            s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        ### END CODE HERE ###
        
        return v, s

    def update_parameters_with_adam(self, grads, t, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-07):
        """
        Update parameters using Adam
        
        Arguments:
        parameters -- python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing your gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        learning_rate -- the learning rate, scalar.
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters -- python dictionary containing your updated parameters 
        v -- Adam variable, moving average of the first gradient, python dictionary
        s -- Adam variable, moving average of the squared gradient, python dictionary
        """
        L = len(self.sizes) -1                 # number of layers in the neural networks
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary
        
        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
            ### START CODE HERE ### (approx. 2 lines)
            self.v["dW" + str(l+1)] = (beta1 * self.v["dW" + str(l+1)]) + ((1 - beta1) * grads['W' + str(l+1)])
            self.v["db" + str(l+1)] = (beta1 * self.v["db" + str(l+1)]) + ((1 - beta1) * grads['b' + str(l+1)])
            ### END CODE HERE ###

            # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
            ### START CODE HERE ### (approx. 2 lines)
            # print('v["dW" + str(l+1)]', v["dW" + str(l+1)])
            v_corrected["dW" + str(l+1)] = self.v["dW" + str(l+1)] / (1 - beta1**t)
            v_corrected["db" + str(l+1)] = self.v["db" + str(l+1)] / (1 - beta1**t)
            ### END CODE HERE ###

            # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
            ### START CODE HERE ### (approx. 2 lines)
            self.s["dW" + str(l+1)] = (beta2 * self.s["dW" + str(l+1)]) + ((1 - beta2) * (grads['W' + str(l+1)]**2))
            self.s["db" + str(l+1)] = (beta2 * self.s["db" + str(l+1)]) + ((1 - beta2) * (grads['b' + str(l+1)]**2))
            ### END CODE HERE ###

            # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
            ### START CODE HERE ### (approx. 2 lines)
            s_corrected["dW" + str(l+1)] = self.s["dW" + str(l+1)] / (1 - beta2**t)
            s_corrected["db" + str(l+1)] = self.s["db" + str(l+1)] / (1 - beta2**t)
            ### END CODE HERE ###

            # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
            ### START CODE HERE ### (approx. 2 lines)
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - learning_rate * (v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon))
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - learning_rate * (v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon))
            # input()
            ### END CODE HERE ###

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []
        loss = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))      

            log = np.log(output, out=np.zeros_like(output), where=(output!=0))
            loss.append(-np.sum(y * log))
        
        return np.mean(predictions), np.mean(loss)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        accuracy_list = []
        t = 0
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)

                if self.optimalizer == 'GD':
                    self.update_network_parameters(changes_to_w) 
                elif self.optimalizer == 'momentum':
                    self.update_parameters_with_momentum(grads=changes_to_w, learning_rate=self.l_rate) 
                elif self.optimalizer == 'adam':
                    t = t + 1
                    self.update_parameters_with_adam(grads=changes_to_w, t=t, learning_rate=self.l_rate)

            accuracy, loss = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%, Loss: {3:.4f}, Learning rate {4}'.format(
                iteration+1, time.time() - start_time, accuracy * 100, loss, self.l_rate
            ))
            accuracy_list.append(accuracy)

            if len(accuracy_list) > 2 and accuracy_list[len(accuracy_list) - 1] - accuracy_list[len(accuracy_list) - 2] < 0.01:
                self.l_rate /= 10
            
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

# dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10], l_rate=0.01, epochs=10, optimalizer='GD')
dnn = DeepNeuralNetwork(sizes=[784, 1024, 10], l_rate=0.01, epochs=20, optimalizer='momentum')
dnn.train(x_train, y_train, x_val, y_val)

acc, loss = dnn.compute_accuracy(x_test, y_test)
print("Test Accuracy: {:.2f}%, Loss: {:.4f}".format(acc * 100, loss))