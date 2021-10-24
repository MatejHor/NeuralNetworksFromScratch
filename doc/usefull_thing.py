import numpy as np

x = np.array([[0, 1], [2,3], [4,5]])
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))    
    return s
print("sigmoid(x) = " + str(sigmoid(x)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s* (1 - s)    
    return ds
print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

def softmax(x):
    x_exp = np.exp(x)
    # print('exp', x_exp)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) 
    # print('sum',x_sum)
    s = x_exp / x_sum
    return s
print("softmax(x) = " + str(softmax(x)))

def softmax_derivation(x):
    s = softmax(x)
    ds = s * (1 - s)
    return ds
print("softmax_derivation(x) = " + str(softmax_derivation(x)))

def ReLu(x):
    func = lambda x: 0.0 if x <= 0.0 else x
    return np.vectorize(func)(x)
print("ReLu(x) = " + str(ReLu(x)))

def ReLu_derivation(x):
    func = lambda x: 0.0 if x <= 0.0 else 1.0
    return np.vectorize(func)(x)
print("ReLu_derivation(x) = " + str(ReLu_derivation(x)))


def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    
    # Divide x by its norm.
    x = x / x_norm
    ### END CODE HERE ###

    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))

# GRADED FUNCTION: softmax
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum

    ### END CODE HERE ###
    
    return s

# GRADED FUNCTION: L1
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.abs(y-yhat))
    ### END CODE HERE ###
    
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

# GRADED FUNCTION: L2
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.dot(abs(y-yhat), abs(y-yhat)))
    ### END CODE HERE ###
    
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))

# Create image from list
from PIL import Image
data = [0,0,0,0,0,0,0,9,8,0,0,34,29,7,0,11,24,0,0,3,3,1,0,1,1,0,0,0,0,0,4,0,0,1,0,0,0,0,0,44,88,99,122,123,80,0,0,0,0,1,1,1,0,0,0,0,0,0,1,2,0,0,0,3,46,174,249,67,0,94,210,61,14,212,157,37,0,0,0,0,1,0,0,0,0,0,2,2,0,23,168,206,242,239,238,214,125,61,113,74,133,236,238,236,203,184,20,0,1,0,0,0,0,0,1,0,0,175,245,223,207,205,206,216,255,237,251,232,223,212,200,205,216,249,173,0,0,2,0,0,0,0,7,0,53,225,201,197,200,201,206,199,197,185,194,204,232,226,249,219,194,205,229,33,0,1,0,0,0,0,1,0,133,223,208,192,195,233,226,216,191,210,188,236,186,0,50,234,207,208,231,133,0,0,0,0,0,0,0,0,216,218,216,194,229,172,64,219,201,200,200,247,68,72,54,165,237,212,219,226,0,0,0,0,0,0,0,50,221,207,220,211,207,165,138,205,192,191,190,232,119,113,67,173,237,217,208,221,29,0,0,0,0,0,0,131,216,200,219,207,212,231,226,193,214,224,206,203,230,122,112,234,224,214,204,224,123,0,0,0,0,0,0,195,212,204,211,203,205,200,184,213,162,138,193,207,203,231,245,208,220,211,203,219,179,0,0,0,0,0,8,185,191,218,233,219,201,221,213,246,114,127,80,129,232,198,218,207,236,227,220,216,172,21,0,0,0,0,21,4,5,64,160,224,224,144,187,197,211,207,186,192,210,212,218,225,236,177,106,56,28,1,0,0,0,0,1,1,0,2,0,116,252,96,120,51,73,70,123,79,76,64,162,252,118,1,3,0,4,2,0,0,0,0,0,0,0,0,0,115,226,145,170,155,165,161,159,125,175,140,174,236,95,0,2,2,0,0,0,0,0,0,0,0,1,2,0,131,225,204,217,221,220,217,224,231,226,237,203,237,102,0,4,2,1,2,0,0,0,0,1,1,0,3,0,135,223,201,199,194,198,195,198,192,203,199,207,231,112,0,4,0,0,0,0,0,0,0,1,1,0,1,0,134,223,199,206,199,201,200,203,206,207,210,206,227,119,0,3,0,0,1,0,0,0,0,0,0,0,1,0,139,223,198,204,200,201,200,201,204,206,208,206,229,128,0,4,0,0,0,0,0,0,0,0,0,0,1,0,145,223,195,205,201,201,200,204,204,206,211,205,230,139,0,2,0,0,0,0,0,0,0,1,0,1,0,0,157,221,194,204,204,201,201,203,205,208,211,204,230,148,0,2,0,1,1,0,0,0,0,1,1,1,0,0,166,220,194,203,203,205,203,203,206,207,212,204,230,157,0,2,1,1,1,0,0,0,0,0,0,0,0,0,171,221,195,206,200,199,203,203,205,206,207,204,226,181,0,0,0,0,0,0,0,0,0,0,0,1,0,0,165,224,197,201,208,199,204,205,207,210,213,207,229,187,0,1,2,0,0,0,0,0,0,0,0,0,0,0,128,201,203,201,207,211,203,205,206,210,213,205,225,191,0,0,2,0,0,0,0,0,0,0,0,1,1,0,141,201,191,188,194,187,187,191,193,195,199,199,218,161,0,0,0,0,0,0,0,0,0,0,0,0,1,0,212,240,213,239,233,239,231,232,236,242,245,224,245,234,0,3,0,0,0,0,0,0,0,0,0,0,0,0,37,69,94,123,127,138,138,142,145,135,125,103,87,56,0,0,0,0,0,0,0]
im2 = Image.new('L', [28, 28])
im2.putdata(data)




