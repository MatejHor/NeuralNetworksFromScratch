#include "./matrix/Matrix.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./dataset/Dataset.cpp"
#include <map>
#include <stdlib.h> // RAND
// comments with lowerCase must be removed!

// map<string, *> cache;
static double layer_dims[] = {784, 255, 255, 200, 10};
static int len_layer = sizeof(layer_dims-1);
static double learning_rate = 0.0075;
static int num_iterations = 3000;

Matrix *A_cache = new Matrix[len_layer];
Matrix *Z_cache = new Matrix[len_layer];

Matrix* initializeBias(){
    Matrix* bias = new Matrix[len_layer];
    for (int layer=1; layer < len_layer; layer++) {
        bias[layer] = Matrix(layer_dims[layer], 1, 0.0);
    }
    return bias;
}

Matrix* initializeWeights(){
    Matrix* weights = new Matrix[len_layer];
    for (int layer=1; layer < len_layer; layer++) {
        weights[layer] = Matrix(layer_dims[layer], layer_dims[layer - 1], 0.01);
    }
    return weights;
}

Matrix* forwardPropagation(Dataset train, Matrix* weights, Matrix* bias) {
    Matrix A = train.getX();
    Matrix A_prev;
    for (int hidden_layer = 0; hidden_layer < (len_layer - 1); hidden_layer++) {
        A_prev = A;
        // linear forward
        Matrix Z = *sum( *dot(weights[hidden_layer], A_prev) , bias[hidden_layer]);
        A_cache[hidden_layer] = A_prev; // cache

        // linear_activation_forward
        A = *reLu(Z);
    }

    Matrix Z = *sum( *dot(weights[len_layer - 1], A_prev) , bias[len_layer - 1]);
    A_cache[len_layer - 1] = A_prev; // cache

    // linear_activation_forward
    Matrix* AL = sigmoid(Z);
    return AL;
}

double computeCostCrossEntropy(Matrix AL, Matrix Y) {
    int m = Y.getColumns();
    double sum_matrix = sumMatrix(
        *sum(
            *multiply(
                Y, 
                *log(AL)), 
            *multiply(
                *subtrack(1,Y), 
                *log(*subtrack(1,AL)))
            )
        );
    return (-1/m) * sum_matrix;
}


int main() {
    // 0. LOAD DATASET
    Dataset train = Dataset(
        "../data/fashion_mnist_train_vectors.csv",
        "../data/fashion_mnist_train_labels.csv"
        );

    // 1.INITIALIZE PARAMETERS 
    Matrix *weigths = initializeWeights();
    Matrix *bias = initializeBias();
    
    // 2. LOOP
    for (int iteration = 0; iteration < num_iterations; iteration++) {
    // 2.1 FORWARD prop
        Matrix AL = *forwardPropagation(train, weigths, bias);
        
    }

    return 0;
}