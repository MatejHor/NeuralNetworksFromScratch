# 1. Define the neural network structure ( # of input units,  # of hidden units, etc). 

# 2. Initialize the model's parameters with random values

# 3. Loop:

## Implement forward propagation -> A_i(last), cache
- compute Z_i = X/Z_i-1 * W_i(vector) + b_i
- activation function A_i = tanh/sigmoid/ReLu...(Z_i)
- save cache data (Z_i, A_i ....)

## Compute loss -> cost
- cross_entropy  
    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m

## Implement backward propagation to get the gradients -> dW_i, db_i (every)
- get params W_i, b_i
- get cache A_i
- dZ_i = A_i - Y(only for out layer)/ np.dot( (W_i+1).T, dZ_i+1) * (1 - A_i(vector)^2)
- dW_i = 1 / m * (np.dot(dZ_i, (A_i-1).T))
- db_i = 1 / m * np.sum(dZ_i, axis=1, keepdims=True)
- save derivation

* (np.sum([[1, 2], [3, 4]], axis=1, keepdims=True) -> [[3], [7]])
* (m - shape of input)

## Update parameters (gradient descent) -> W_i, b_i (every)
- get params W_i, b_i (forwardprop)
- get params dW_i, db_i (backprop)
- learning_rate (alpha=1.2)
- W_i = W_1 - learning_rate * dW_i
- b_i = b_1 - learning_rate * db_i
- save params W_i, b_i

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
    - aktivacne funkcie sigmoid/softmax/tanh?/ReLu
- function forward_propagation(X, weights)
- function back_propagation( weights, cache, X, Y))
- function loss_function(A_i(last), Y, weights) cross_entropy_loss
- function update_weights(weights, derivation)
- function initialize_parameters(n_x, n_h, n_y) random/he
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

## INTEGRATION 
model(  
    X(input),   
    Y(true_labels),   
    n_h(size_of_hidden_layer),   
    num_iterations   
    )  

1. initialize_parameters(n_x, n_h, n_y) -> weights
2. Training LOOP -> weights
- A_i(last), cache = forward_propagation()
- cost = compute_cost()
- grads = backward_propagation()
- weights = update_weights()
3. Prediction(weights, X) -> Y
- A_i(last), cache = forward_propagation()
- Y = round( A_i(last) )