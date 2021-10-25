# NOTES
- class pre Dataset (vstup)
    - y - pole lablov
    - Matrix (X) - transponovana matica vstupnych dat
    - shape pre X a y
    - print pre vypis (adresu premennej)
    - getter a setter pre X aj y
- class pre Matrix
    - pole poli
    - .T transponovanie matice
- class pre Operation
    - sum nad maticou
    - sum nad dimenziami matice
    - dot nad maticou
    - multiple nad maticou
    - log nad maticou
    - aktivacne funkcie sigmoid/softmax/ReLu
- function forward_propagation(X, weights)
- function back_propagation( weights, cache, X, Y))
- function loss_function(A_i(last), Y, weights) cross_entropy_loss
- function update_weights(weights, derivation)
- function initialize_parameters(n_x, n_h, n_y) random/he

# INTEGRATION 
1. Initialize parameters -> parameters
### initialize_parameters(layer_dims) -> parameters
    layer dims list containing the dimension of each layer
    
input layer        | hidden layer(dense)| hidden layer(dense)| output layer
784                | 255                | 200                | 10
                   | 255rows 784columns | 200rows 255columns | 10rows 200columns  | 1rows 10columns

    for layer_i in layers
        W_i = random matrix [len(W_i)][len(W_i-1)] * 0.01
        b_i = random double [len(W_i)]
    
2. Training LOOP -> weights
### forward_propagation(X, parametre) -> AL(last post-activation value), cache(W, A_prev, b, Z)
    - A = X forloop len(layers)
        - A_prev = A
        - Z = np.dot(W, A) + b, cache(W, A, b)
        - A = activationFunction(Z)(relu and last sigmoid), cache(W, A_prev, b, Z) 
    - AL, cache

### compute_cost(AL, Y) -> cost(value) (cross-entropy)
    - m = y.shape
    - cost = -1/m * np.sum( np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)) )
    - np.squeeze(cost) (only remove useless dimension not needed!)

### backward_propagation(Y, AL, caches) -> dW_temp,db_temp, dA_prev_temp every
    - dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    - najprv len sigmoid
    - forloop reverse len(layer)
        - current cache current layer (Z, W, A_prev)
        - dZ = relu_backward(dA, activation_cache)
        - m = A_prev.shape[1]
        - dW_temp = (1/m) * np.dot(dZ, A_prev.T)
        - db_temp = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        - dA_prev_temp = np.dot(W.T, dZ)

### update_weights(parameters, grads, learning_rate) -> parameters
    - forloop len(layers/2???)
        - params[W_i] = params[W_i] - learningRate * params[dW_i]
        - params[b_i] = params[b_i] - learningRate * params[db_i]
3. Prediction -> Y
### predict(parameters, X) -> predictions
- A = forward_propagation(X, parameters)
- predictions = np.round(A2)

## SUMARIZE WORKFLOW (X, Y, layers_dims, learning_rate, num_iterations) "MAIN"
layers_dims = [784, 255, 255, 100, 10] // 3 Skryte vrstvy
learning_rate = 0.0075
num_iterations = 3000

- parameters = initialize_parameters_deep(layers_dims)
- forloop len(num_iterations) -> parameters
    - AL <- forward_propagation(X, parameters) uchovame **cache**
    - cost <- compute_cost(AL, Y)
    - gradas(3 derivation per layer) <- backward_propagation(AL, Y, cache)
    - parameters <- update_parameters(parameters, grads, learning_rate)

    - print ("Cost after iteration %d: %f" %(i, cost))

- predictions = predict(X_test, parameters)
- print("ACC %f", accuracy(X_test, Y_test))