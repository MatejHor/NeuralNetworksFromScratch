#include "./matrix/Matrix.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./dataset/Dataset.cpp"
#include <stdlib.h> // RAND

// SIZE of double[]
#include <bits/stdc++.h>

#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>

// TIME
#include <chrono>
using namespace chrono;

static double layer_dims[] = {784, 255, 200, 10};
static int len_layer = (*(&layer_dims + 1) - layer_dims) - 1;
static double learning_rate = 0.0075;
static int num_iterations = 100;
bool VERBOSE = true;

Matrix *AL = NULL;
vector<Matrix *> A_cache;
vector<Matrix *> dA_cache;
vector<Matrix *> dW_cache;
vector<Matrix *> db_cache;

vector<Matrix *> initializeBias()
{
    vector<Matrix *> bias;
    for (int layer = 0; layer < len_layer; layer++)
    {
        bias.push_back(new Matrix(layer_dims[layer + 1], 1, 0.0));
    }
    return bias;
}

vector<Matrix *> initializeWeights()
{
    vector<Matrix *> weights;
    for (int layer = 0; layer < len_layer; layer++)
    {
        weights.push_back(new Matrix(layer_dims[layer + 1], layer_dims[layer], 0.01));
    }
    return weights;
}

Matrix *forwardPropagation(Matrix *X, vector<Matrix *> weights, vector<Matrix *> bias)
{
    Matrix *A_prev = X;
    Matrix *Z = NULL;
    Matrix *dot_W_A_prev;

    for (int hidden_layer = 0; hidden_layer < (len_layer - 1); hidden_layer++)
    {
        dot_W_A_prev = dot(weights.at(hidden_layer), A_prev);
        Z = sumVector(dot_W_A_prev, bias.at(hidden_layer));

        A_cache.push_back(A_prev);
        A_prev = reLu(Z);

        // Destruct computed values
        dot_W_A_prev->~Matrix();
        Z->~Matrix();
    }
    dot_W_A_prev = dot(weights.at(len_layer - 1), A_prev);
    Z = sumVector(dot_W_A_prev, bias.at(len_layer - 1));

    A_cache.push_back(A_prev);
    A_prev = softmax(Z);

    // Destruct computed values
    dot_W_A_prev->~Matrix();
    Z->~Matrix();
    return A_prev;
}

double computeCostCrossEntropy(Matrix *AL, Matrix *Y)
{
    int m = Y->getColumns();
    Matrix *log_AL = log(AL);
    Matrix *multiply_Y_log_AL = multiply(Y, log_AL);

    Matrix *subtrack_Y = subtrack(1, Y);
    Matrix *subtrack_AL = subtrack(1, AL);
    Matrix *logSubtrack_AL = log(subtrack_AL);
    Matrix *multiplySubtrack_Y_logSubtrack_AL = multiply(subtrack_Y, logSubtrack_AL);

    Matrix *sumMatrix_ = sum(multiply_Y_log_AL, multiplySubtrack_Y_logSubtrack_AL);
    double sumMatrixVal = sumMatrix(sumMatrix_);

    // Destruct computed values
    log_AL->~Matrix();
    multiply_Y_log_AL->~Matrix();

    subtrack_Y->~Matrix();
    subtrack_AL->~Matrix();
    logSubtrack_AL->~Matrix();
    multiplySubtrack_Y_logSubtrack_AL->~Matrix();

    sumMatrix_->~Matrix();

    return (-1.0 / m) * sumMatrixVal;
}

void backwardPropagation(Matrix *AL, Matrix *Y, vector<Matrix *> weights)
{
    double m = AL->getColumns();

    Matrix *divide_Y_AL = divide(Y, AL);
    Matrix *subtrack_Y = subtrack(1, Y);
    Matrix *subtrack_AL = subtrack(1, AL);
    Matrix *divideSubtrack_Y_AL = divide(subtrack_Y, subtrack_AL);
    Matrix *subtrackDivide_Y_AL_DivideSubtrack_Y_AL = subtrack(divide_Y_AL, divideSubtrack_Y_AL);
    Matrix *dAL = multiply(subtrackDivide_Y_AL_DivideSubtrack_Y_AL, -1);

    // Destruct computed values
    divide_Y_AL->~Matrix();
    subtrack_Y->~Matrix();
    subtrack_AL->~Matrix();
    divideSubtrack_Y_AL->~Matrix();
    subtrackDivide_Y_AL_DivideSubtrack_Y_AL->~Matrix();

    Matrix *dZ = softmaxDerivation(dAL);

    Matrix *trans_A_cache = (A_cache.at(len_layer - 1))->T();
    Matrix *dot_dZ_AprevT = dot(dZ, trans_A_cache);
    Matrix *dW = multiply(dot_dZ_AprevT, (1.0 / m));
    Matrix *sumDimension_dZ = sumDimension(dZ);
    Matrix *db = multiply(sumDimension_dZ, (1.0 / m));
    Matrix *trans_weight = (weights.at(len_layer - 1))->T();
    Matrix *dA_prev = dot(trans_weight, dZ);

    dA_cache.push_back(dA_prev);
    dW_cache.push_back(dW);
    db_cache.push_back(db);

    // Destruct computed values
    dAL->~Matrix();
    dZ->~Matrix();
    trans_A_cache->~Matrix();
    dot_dZ_AprevT->~Matrix();
    sumDimension_dZ->~Matrix();
    trans_weight->~Matrix();

    // cout << "[+] LOOPING [+]" << endl;
    for (int hidden_layer = (len_layer - 2); hidden_layer >= 0; hidden_layer--)
    {
        // cout << "Hidden layer " << hidden_layer << endl;
        dZ = reLuDerivation(dA_prev);

        trans_A_cache = (A_cache.at(hidden_layer))->T();
        dot_dZ_AprevT = dot(dZ, trans_A_cache);
        dW = multiply(dot_dZ_AprevT, (1.0 / m));
        sumDimension_dZ = sumDimension(dZ);
        db = multiply(sumDimension_dZ, (1.0 / m));
        trans_weight = (weights.at(hidden_layer))->T();
        dA_prev = dot(trans_weight, dZ);

        dA_cache.push_back(dA_prev);
        dW_cache.push_back(dW);
        db_cache.push_back(db);

        // Destruct computed values
        dZ->~Matrix();
        trans_A_cache->~Matrix();
        dot_dZ_AprevT->~Matrix();
        sumDimension_dZ->~Matrix();
        trans_weight->~Matrix();
    }
}

void updateParametersGradientDescend(vector<Matrix *> &bias, vector<Matrix *> &weights)
{
    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        Matrix *multiplydBLearningRate = multiply(db_cache.at(len_layer - (hidden_layer + 1)), learning_rate);
        Matrix *old_bias = bias.at(hidden_layer);
        bias.at(hidden_layer) = subtrack(old_bias, multiplydBLearningRate);

        // Destruct computed values
        old_bias->~Matrix();
        multiplydBLearningRate->~Matrix();

        Matrix *multiplydWLearningRate = multiply(dW_cache.at(len_layer - (hidden_layer + 1)), learning_rate);
        Matrix *old_weights = weights.at(hidden_layer);
        weights.at(hidden_layer) = subtrack(old_weights, multiplydWLearningRate);

        // Destruct computed values
        old_weights->~Matrix();
        multiplydWLearningRate->~Matrix();
    }
}

double predict(Dataset *data, vector<Matrix *> weights, vector<Matrix *> bias, string measure, bool verbose)
{
    Matrix *A_prev = data->getX();
    Matrix *A_free = NULL;
    Matrix *Z = NULL;
    Matrix *dot_W_A_prev;

    auto pred_start = high_resolution_clock::now();
    for (int layer = 0; layer < (len_layer - 1); layer++)
    {
        dot_W_A_prev = dot(weights.at(layer), A_prev);
        Z = sumVector(dot_W_A_prev, bias.at(layer));
        A_free = A_prev;
        A_prev = reLu(Z);

        // Destruct
        {
            dot_W_A_prev->~Matrix();
            Z->~Matrix();
            if (layer != 0)
                A_free->~Matrix();
        }
    }
    dot_W_A_prev = dot(weights.at(len_layer - 1), A_prev);
    Z = sumVector(dot_W_A_prev, bias.at(len_layer - 1));
    Matrix *A_predict = softmax(Z);
    auto pred_stop = high_resolution_clock::now();

    // Destruct
    {
        A_prev->~Matrix();
        dot_W_A_prev->~Matrix();
        Z->~Matrix();
    }

    Matrix *PrediLabel = squeeze(A_predict, "max");
    double measureVal;
    if (measure.compare("accuracy") == 0)
        measureVal = data->accuracy(PrediLabel);
    else
        measureVal = data->f1_mikro(PrediLabel);

    if (verbose)
        cout << "Predict Time " << duration_cast<milliseconds>(pred_stop - pred_start).count() << " milliseconds" << endl;
    if (verbose)
        cout << measure << ": " << measureVal << endl;

    // destruct
    {
        PrediLabel->~Matrix();
        A_predict->~Matrix();
    }
    return measureVal;
}

void saveParameters(vector<Matrix *> weights, vector<Matrix *> bias)
{
    ofstream weights_file;
    ofstream bias_file;
    weights_file.open("weights.txt");
    bias_file.open("bias.txt");

    for (int layer = 0; layer < len_layer; layer++)
    {
        string weights_value = "";
        string bias_value = "";
        double **matrix_weights = weights.at(layer)->getMatrix();
        double **matrix_bias = bias.at(layer)->getMatrix();
        int rows = weights.at(layer)->getRows();
        int columns = weights.at(layer)->getColumns();

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
        weights_file << "Weigth layer {" << layer << "}\n"
                     << weights_value << endl
                     << endl;
        bias_file << "Bias layer {" << layer << "}\n"
                  << bias_value << endl
                  << endl;
    }
    weights_file.close();
    bias_file.close();
}

void freeParameters(vector<Matrix *> &weights, vector<Matrix *> &bias)
{
    for (int i = 0; i < len_layer; i++)
    {
        bias.at(i)->~Matrix();
        weights.at(i)->~Matrix();
    }
    bias.clear();
    weights.clear();
}

void freeCache()
{
    for (int layer = 0; layer < len_layer; layer++)
    {
        if (layer != 0)
            A_cache.at(layer)->~Matrix();
        dA_cache.at(layer)->~Matrix();
        dW_cache.at(layer)->~Matrix();
        db_cache.at(layer)->~Matrix();
    }
    A_cache.clear();
    dA_cache.clear();
    dW_cache.clear();
    db_cache.clear();
}

int main()
{
    cout << "0. LOAD DATASET" << endl;
    Dataset train = Dataset(
        "../data/fashion_mnist_train_vectors.csv",
        "../data/fashion_mnist_train_labels.csv",
        100,
        VERBOSE);

    cout << "1. INITIALIZE PARAMETERS" << endl;
    vector<Matrix *> weights = initializeWeights();
    vector<Matrix *> bias = initializeBias();

    cout << "2. LOOP " << endl;
    auto start = high_resolution_clock::now();
    for (int iteration = 0; iteration < num_iterations; iteration++)
    {
        if (AL != NULL)
            AL->~Matrix();
        // cout << "2.1 FORWARD PROPAGATION" << endl;
        AL = forwardPropagation(train.getX(), weights, bias);

        // cout << "2.2 COMPUTE COST" << endl;
        double cost = computeCostCrossEntropy(AL, train.getY());

        // cout << "2.3 BACKWARD PROPAGATION" << endl;
        backwardPropagation(AL, train.getY(), weights);

        // cout << "2.4 UPDATE PARAMETERS" << endl;
        updateParametersGradientDescend(bias, weights);

        if (iteration % 10 == 0)
        // if (true)
        // if (iteration % 50 == 0 || true)
        {
            cout << "Cost after iteration " << iteration << ": " << cost << endl;
        }

        // cout << "2.5 FREE CACHE" << endl;
        freeCache();
    }
    auto stop = high_resolution_clock::now();
    if (VERBOSE) cout << "Time for training " << duration_cast<milliseconds>(stop - start).count() << " milliseconds" << endl;

    cout << "3.1 PREDICTION TRAIN" << endl;
    double test_acc = predict(&train, weights, bias, "accuracy", VERBOSE);

    cout << "3.2 PREDICTION TEST" << endl;
    {
        Dataset test = Dataset(
            "../data/fashion_mnist_test_vectors.csv",
            "../data/fashion_mnist_test_labels.csv",
            100,
            VERBOSE);

        double test_acc = predict(&test, weights, bias, "accuracy", VERBOSE);
    }

    if (VERBOSE) cout << "4. SAVE PARAMETERS" << endl;
    saveParameters(weights, bias);

    if (VERBOSE) cout << "5. FREE BIAS && WEIGHTS VECTOR" << endl;
    freeParameters(weights, bias);
    return 0;
}