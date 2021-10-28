#include "./matrix/Matrix.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./dataset/Dataset.cpp"
#include <stdlib.h> // RAND

#include <iostream>
#include <fstream>
#include <string.h>

static double layer_dims[] = {784, 255, 225, 200, 10};
static int len_layer = 4;
static double learning_rate = 0.0075;
static int num_iterations = 3000;

Matrix *A_cache = new Matrix[len_layer];
Matrix *Z_cache = new Matrix[len_layer];

Matrix *dA_cache = new Matrix[len_layer];
Matrix *dW_cache = new Matrix[len_layer];
Matrix *db_cache = new Matrix[len_layer];

Matrix *initializeBias()
{
    Matrix *bias = new Matrix[len_layer + 1];
    for (int layer = 0; layer < len_layer; layer++)
    {
        bias[layer] = *(new Matrix(layer_dims[layer + 1], 1, 0.0));
        cout << "Initialize bias " << bias << " (" << layer_dims[layer + 1] << "x" << 1 << ") " << endl;
        // bias[layer].print();
    }
    return bias;
}

Matrix *initializeWeights()
{
    Matrix *weights = new Matrix[len_layer + 1];
    for (int layer = 0; layer < len_layer; layer++)
    {
        weights[layer] = *(new Matrix(layer_dims[layer + 1], layer_dims[layer], 0.01));
        cout << "Initialize weights " << weights << " (" << layer_dims[layer + 1] << "x" << layer_dims[layer] << ") " << endl;
        // weights[layer].print();
    }
    return weights;
}

Matrix *forwardPropagation(Matrix X, Matrix *weights, Matrix *bias)
{
    Matrix A_prev = X;
    for (int hidden_layer = 0; hidden_layer < (len_layer - 1); hidden_layer++)
    {
        A_prev.printParams();
        weights[hidden_layer].printParams();

        Matrix dotm = *dot(weights[hidden_layer], A_prev);
        Matrix Z = *sum(*dot(weights[hidden_layer], A_prev), bias[hidden_layer]);
        A_cache[hidden_layer] = A_prev;
        A_prev = *reLu(Z);
    }

    Matrix Z = *sum(*dot(weights[len_layer - 1], A_prev), bias[len_layer - 1]);
    A_cache[len_layer - 1] = A_prev;
    Matrix *AL = softmax(Z);
    return AL;
}

void backwardPropagation(Matrix AL, Matrix Y, Matrix *weights)
{
    double m = AL.getColumns();
    Matrix dAL = *multiply(
        *subtrack(
            *divide(Y, AL),
            *divide(*subtrack(1, Y), *subtrack(1, AL))),
        -1);

    Matrix dZ = *softmaxDerivation(dAL);
    Matrix dW = *multiply(
        *dot(
            dZ,
            *(A_cache[len_layer - 1].T())),
        (1 / m));
    Matrix db = *multiply(*sumDimension(dZ), (1 / m));
    Matrix dA_prev = *dot(*weights[len_layer - 1].T(), dZ);

    dA_cache[len_layer - 1] = dA_prev;
    dW_cache[len_layer - 1] = dW;
    db_cache[len_layer - 1] = db;

    for (int hidden_layer = (len_layer - 2); hidden_layer >= 0; hidden_layer--)
    {
        Matrix dZ = *reLuDerivation(dAL);
        Matrix dW = *multiply(
            *dot(
                dZ,
                *(A_cache[hidden_layer].T())),
            (1 / m));
        Matrix db = *multiply(*sumDimension(dZ), (1 / m));
        Matrix dA_prev = *dot(*weights[hidden_layer].T(), dZ);

        dA_cache[hidden_layer] = dA_prev;
        dW_cache[hidden_layer] = dW;
        db_cache[hidden_layer] = db;
    }
}

Matrix *updateWeights(Matrix *weights)
{
    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        weights[hidden_layer] = *subtrack(
            weights[hidden_layer],
            *multiply(dW_cache[hidden_layer], learning_rate));
    }
    return weights;
}

Matrix *updateBias(Matrix *bias)
{
    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        bias[hidden_layer] = *subtrack(
            bias[hidden_layer],
            *multiply(db_cache[hidden_layer], learning_rate));
    }
    return bias;
}

double computeCostCrossEntropy(Matrix AL, Matrix Y)
{
    int m = Y.getColumns();
    double sum_matrix = sumMatrix(
        *sum(
            *multiply(
                Y,
                *log(AL)),
            *multiply(
                *subtrack(1, Y),
                *log(*subtrack(1, AL)))));
    return (-1 / m) * sum_matrix;
}

Matrix *predict(Matrix X, Matrix *weights, Matrix *bias)
{
    return forwardPropagation(X, weights, bias);
}

double f1_mikro(Matrix Y, Matrix AL)
{
    double **y = Y.getMatrix();
    double **al = AL.getMatrix();

    double TP = 0;
    double TN = 0;
    double FP = 0;
    double FN = 0;

    for (int i = 0; i < 10; i++)
    {
        for (int row = 0; row < Y.getRows(); row++)
        {
            (al[row][0] == i && y[row][0] == i && ++TP);
            (al[row][0] != i && y[row][0] != i && ++TN);
            (al[row][0] == i && y[row][0] != i && ++FP);
            (al[row][0] != i && y[row][0] == i && ++FN);
        }
    }

    double precision = TP / (TP + FP);
    double recall = TP / (TP + FN);
    return 2 * ((precision * recall) / (precision + recall));
}

void saveParameters(Matrix *weights, Matrix *bias)
{
    ofstream weights_file;
    ofstream bias_file;
    weights_file.open("weights.txt");
    bias_file.open("weights.txt");

    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        string weights_value = "";
        string bias_value = "";
        double **matrix_weights = weights[hidden_layer].getMatrix();
        double **matrix_bias = bias[hidden_layer].getMatrix();
        int rows = weights[hidden_layer].getRows();
        int columns = weights[hidden_layer].getColumns();

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                weights_value += to_string(matrix_weights[row][column]);
            }
            bias_value += to_string(matrix_bias[row][0]);

            weights_value += "\n";
            bias_value += "\n";
        }
        weights_file << weights_value << endl
                     << endl;
        bias_file << bias_value << endl
                  << endl;
    }
    weights_file.close();
    bias_file.close();
}

int main()
{
    cout << "0. LOAD DATASET" << endl;
    Dataset train = Dataset(
        "../data/fashion_mnist_sample_vectors.csv",
        "../data/fashion_mnist_sample_labels.csv");

    // Dataset train = Dataset(
    //     "../data/fashion_mnist_train_vectors.csv",
    //     "../data/fashion_mnist_train_labels.csv");

    cout << "1. INITIALIZE PARAMETERS" << endl;
    Matrix *weights = initializeWeights();
    Matrix *bias = initializeBias();

    cout << "2. LOOP" << endl;
    for (int iteration = 0; iteration < num_iterations; iteration++)
    {
        cout << "2.1 FORWARD PROPAGATION" << endl;
        Matrix AL = *forwardPropagation(train.getX(), weights, bias);

        cout << "2.2 COMPUTE COST" << endl;
        double cost = computeCostCrossEntropy(AL, train.getY());

        cout << "2.3 BACKWARD PROPAGATION" << endl;
        backwardPropagation(AL, train.getY(), weights);

        cout << "2.4 UPDATE PARAMETERS" << endl;
        weights = updateWeights(weights);
        bias = updateBias(bias);

        if (iteration % 100 == 0)
        {
            cout << "Cost after iteration " << iteration << ": " << cost << endl;
        }
    }
    // Matrix result = *predict(train.getX(), weights, bias);
    // cout << "F1_Mikro: " << f1_mikro(train.getY(), result);

    // cout << "3. PREDICTION" << endl;
    // Dataset test = Dataset(
    //     "../data/fashion_mnist_test_vectors.csv",
    //     "../data/fashion_mnist_test_labels.csv");

    // Matrix result = *predict(test.getX(), weights, bias);
    // cout << "F1_Mikro: " << f1_mikro(test.getY(), result);

    // cout << "4. SAVE PARAMETERS" << endl;
    // saveParameters(weights, bias);

    cout << "-1. FREE BIAS & WEIGHTS" << endl;
    for (int i = 0; i < len_layer; i++)
    {
        // cout << &bias[i] << " (" << bias[i].getRows() << ", " << bias[i].getColumns() << ")" << endl;
        // cout << &weights[i] << " (" << weights[i].getRows() << ", " << weights[i].getColumns() << ")" << endl;
        bias[i].~Matrix();
        weights[i].~Matrix();
    }
    return 0;
}