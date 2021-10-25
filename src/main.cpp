#include "./matrix/Matrix.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./dataset/Dataset.cpp"
#include <map>
#include <stdlib.h> // rand

// map<string, *> cache;
static double layer_dims[] = {784, 255, 255, 200, 10};
static int len_layer = sizeof(layer_dims-1);
static double learning_rate = 0.0075;
static int num_iterations = 3000;

Matrix** initializeBias(){
    Matrix* bias = new Matrix[len_layer];
    for (int layer=1; layer < len_layer; layer++) {
        bias[layer] = Matrix(layer_dims[layer], 1, 0.0);
    }
    return &bias;
}

Matrix** initializeWeights(){
    Matrix* weights = new Matrix[len_layer];
    for (int layer=1; layer < len_layer; layer++) {
        weights[layer] = Matrix(layer_dims[layer], layer_dims[layer - 1], 0.01);
    }
    return &weights;
}

Matrix* forwardPropagation(Dataset train, Matrix** weights, Matrix** bias) {
    Matrix A = train.getX();
    for (int hidden_layer = 0; hidden_layer < (len_layer - 1); hidden_layer++) {
        
    }
}

int main() {
    // 0. LOAD DATASET
    Dataset train = Dataset(
        "../data/fashion_mnist_train_vectors.csv",
        "../data/fashion_mnist_train_labels.csv"
        );

    // 1.INITIALIZE PARAMETERS 
    Matrix **weigths = initializeWeights();
    Matrix **bias = initializeBias();
    
    Matrix *A_cache = new Matrix[len_layer];
    Matrix *Z_cache = new Matrix[len_layer];
    // 2. LOOP
    for (int iteration = 0; iteration < num_iterations; iteration++) {
    // 2.1 FORWARD prop
        
           
    }

    return 0;
}