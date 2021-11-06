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

static int len_layer;
static double learning_rate;
static int epochs;
static int batchSize;
double accuracy;
bool VERBOSE;

Matrix *AL = NULL;
vector<Matrix *> A_cache;
vector<Matrix *> Z_cache;
vector<Matrix *> dA_cache;
vector<Matrix *> dW_cache;
vector<Matrix *> db_cache;

vector<Matrix *> initializeBias(double layer_dims[])
{
    vector<Matrix *> bias;
    for (int layer = 0; layer < len_layer; layer++)
    {
        bias.push_back(new Matrix(layer_dims[layer + 1], 1, 0.0));
    }
    return bias;
}

vector<Matrix *> initializeRandomWeights(double layer_dims[])
{
    vector<Matrix *> weights;
    for (int layer = 0; layer < len_layer; layer++)
    {
        weights.push_back(new Matrix(layer_dims[layer + 1], layer_dims[layer], 0.01));
    }
    return weights;
}

vector<Matrix *> initializeHeWeights(double layer_dims[])
{
    vector<Matrix *> weights;
    for (int layer = 0; layer < len_layer; layer++)
    {
        double he = sqrt(1. / layer_dims[layer + 1]);
        weights.push_back(new Matrix(layer_dims[layer + 1], layer_dims[layer], he));
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
    // Formula Z = dot(W[0], A[0-1] a.k.a X) + b[0]
        Z = sumVector(dot_W_A_prev, bias.at(hidden_layer));

        A_cache.push_back(A_prev);
        Z_cache.push_back(Z);
    // Formula A[0] = activation(Z)
        A_prev = reLu(Z);

        {
            dot_W_A_prev->~Matrix();
        }
    }
    dot_W_A_prev = dot(weights.at(len_layer - 1), A_prev);
// Formula Z = dot(W[last], A[last - 1]) + b[last]
    Z = sumVector(dot_W_A_prev, bias.at(len_layer - 1));

    A_cache.push_back(A_prev);
    Z_cache.push_back(Z);

// Formula A[last] = activation(Z)
    A_prev = softmax(Z);

    {
        dot_W_A_prev->~Matrix();
    }
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

    {
        log_AL->~Matrix();
        multiply_Y_log_AL->~Matrix();

        subtrack_Y->~Matrix();
        subtrack_AL->~Matrix();
        logSubtrack_AL->~Matrix();
        multiplySubtrack_Y_logSubtrack_AL->~Matrix();

        sumMatrix_->~Matrix();
    }
    return -(sumMatrixVal / m);
    // return (-1.0 / m) * sumMatrixVal;
}

void backwardPropagation(Matrix *AL, Matrix *Y, vector<Matrix *> weights)
{
    double m = AL->getColumns();

    // Formula dA[last] = - ((Y / AL ) - ((1 - Y) / (1 - AL)))
    // Matrix *divide_Y_AL = divide(Y, AL);
    // Matrix *subtrack_Y = subtrack(1, Y);
    // Matrix *subtrack_AL = subtrack(1, AL);
    // Matrix *divideSubtrack_Y_AL = divide(subtrack_Y, subtrack_AL);
    // Matrix *subtrackDivide_Y_AL_DivideSubtrack_Y_AL = subtrack(divide_Y_AL, divideSubtrack_Y_AL);
    // Matrix *dAL = multiply(subtrackDivide_Y_AL_DivideSubtrack_Y_AL, -1);
    // {
    //     divide_Y_AL->~Matrix();
    //     subtrack_Y->~Matrix();
    //     subtrack_AL->~Matrix();
    //     divideSubtrack_Y_AL->~Matrix();
    //     subtrackDivide_Y_AL_DivideSubtrack_Y_AL->~Matrix();
    // }

    // EASER VERSION
    // dAL = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
    Matrix *subtrack_Y_AL = subtrack(Y, AL);
    Matrix* dAL = multiply(subtrack_Y_AL, (2.0/Y->getRows()));
    subtrack_Y_AL->~Matrix();

    Matrix *activeDerivation = softmaxDerivation(Z_cache.at(len_layer - 1));
// Formula dZ[last] = dA[last]*activation(Z[last])
    Matrix *dZ = multiply(dAL, activeDerivation);
// Formula dW[last] = (1/m) * dot(dZ, A[last - 1].T)
    Matrix *Aprev_T = (A_cache.at(len_layer - 1))->T();
    Matrix *dot_dZ_AprevT = dot(dZ, Aprev_T);
    Matrix *dW = multiply(dot_dZ_AprevT, (1.0 / m));
// Formula db[last] = (1/m) * sumDimension(dZ)
    Matrix *sumDimension_dZ = sumDimension(dZ);
    Matrix *db = multiply(sumDimension_dZ, (1.0 / m));
// Formula dA[last - 1] = dot(W[last].T, dZ)
    Matrix *W_T = (weights.at(len_layer - 1))->T();
    Matrix *dA = dot(W_T, dZ);

    dW_cache.push_back(dW);
    db_cache.push_back(db);

    {
        dAL->~Matrix();
        activeDerivation->~Matrix();
        dZ->~Matrix();
        Aprev_T->~Matrix();
        dot_dZ_AprevT->~Matrix();
        sumDimension_dZ->~Matrix();
        W_T->~Matrix();
    }

    for (int hidden_layer = (len_layer - 2); hidden_layer >= 0; hidden_layer--)
    {

    // Formula dZ[last - 1] = dA[last]*activation(Z[last- 1])
        Matrix *activeDerivation = reLuDerivation(Z_cache.at(hidden_layer));
        dZ = multiply(dA, activeDerivation);
    // Formula dW[last - 1] = (1/m) * dot(dZ, A[last - 2].T)
        Aprev_T = (A_cache.at(hidden_layer))->T();
        dot_dZ_AprevT = dot(dZ, Aprev_T);
        dW = multiply(dot_dZ_AprevT, (1.0 / m));
    // Formula db[last - 1] = (1/m) * sumDimension(dZ)
        sumDimension_dZ = sumDimension(dZ);
        db = multiply(sumDimension_dZ, (1.0 / m));
    // Formula dA[last - 2] = dot(W[last - 1].T, dZ)
        W_T = (weights.at(hidden_layer))->T();
        dA = dot(W_T, dZ);

        dW_cache.push_back(dW);
        db_cache.push_back(db);

        {
            activeDerivation->~Matrix();
            dZ->~Matrix();
            Aprev_T->~Matrix();
            dot_dZ_AprevT->~Matrix();
            sumDimension_dZ->~Matrix();
            W_T->~Matrix();
        }
    }
}

void updateParametersGradientDescend(vector<Matrix *> &bias, vector<Matrix *> &weights)
{
    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        Matrix *multiplydBLearningRate = multiply(db_cache.at(len_layer - (hidden_layer + 1)), learning_rate);
        Matrix *old_bias = bias.at(hidden_layer);
        bias.at(hidden_layer) = subtrack(old_bias, multiplydBLearningRate);
        {
            old_bias->~Matrix();
            multiplydBLearningRate->~Matrix();
        }

        Matrix *multiplydWLearningRate = multiply(dW_cache.at(len_layer - (hidden_layer + 1)), learning_rate);
        Matrix *old_weights = weights.at(hidden_layer);
        weights.at(hidden_layer) = subtrack(old_weights, multiplydWLearningRate);

        {
            old_weights->~Matrix();
            multiplydWLearningRate->~Matrix();
        }
    }
}

int initializeMiniBatch(vector<Matrix *> &X, vector<Matrix *> &Y, Dataset *data, int batchSize)
{
    Matrix *yData = data->getY();
    Matrix *xData = data->getX();
    int length_data = xData->getColumns();

    while (length_data != 0)
    {
        if (length_data - batchSize < 0)
            batchSize = length_data;
        X.push_back(new Matrix(xData, batchSize));
        Y.push_back(new Matrix(yData, batchSize));

        length_data -= batchSize;
    }
    return X.size();
}

double predict(Dataset *data, vector<Matrix *> weights, vector<Matrix *> bias, string measure, bool verbose)
{
    Matrix *A_prev = data->getX();
    Matrix *A_free = NULL;
    Matrix *Z = NULL;
    Matrix *dot_W_A_prev;

    for (int layer = 0; layer < (len_layer - 1); layer++)
    {
        dot_W_A_prev = dot(weights.at(layer), A_prev);
        Z = sumVector(dot_W_A_prev, bias.at(layer));
        A_prev = reLu(Z);

        {
            dot_W_A_prev->~Matrix();
            Z->~Matrix();
            if (layer != 0)
                A_prev->~Matrix();
        }


    }

    dot_W_A_prev = dot(weights.at(len_layer - 1), A_prev);
    Z = sumVector(dot_W_A_prev, bias.at(len_layer - 1));
    A_prev = softmax(Z);


    Matrix *predictLabel = squeeze(A_prev, "max");
    double measureVal;
    if (measure.compare("accuracy") == 0)
        measureVal = data->accuracy(predictLabel);
    else
        measureVal = data->f1_mikro(predictLabel);

    {
        dot_W_A_prev->~Matrix();
        Z->~Matrix();
        A_prev->~Matrix();
        predictLabel->~Matrix();
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
            weights_value += "{\n";
            bias_value += "{\n";
            for (int column = 0; column < columns; column++)
            {
                weights_value += to_string(matrix_weights[row][column]);
                weights_value += ", ";
            }
            bias_value += to_string(matrix_bias[row][0]);

            weights_value += "}\n";
            bias_value += "}\n";
        }

        weights_file << "Weigth layer [" << layer << "]\n"
                     << rows << " " << columns << "\n{"
                     << weights_value
                     << "}\n\n";
        bias_file << "Bias layer [" << layer << "]\n"
                  << rows << " " << 1 << "\n{"
                  << bias_value
                  << "}\n\n";
    }
    weights_file.close();
    bias_file.close();
}

void freeMatrixVector(vector<Matrix *> &vector1, vector<Matrix *> &vector2, int count)
{
    for (int i = 0; i < count; i++)
    {
        vector1.at(i)->~Matrix();
        vector2.at(i)->~Matrix();
    }
    vector1.clear();
    vector2.clear();
}

void freeVector(vector<Matrix *> &vector)
{
    for (int i = 0; i < vector.size(); i++)
        vector.at(i)->~Matrix();
    vector.clear();
}

void freeCache()
{
    for (int layer = 0; layer < len_layer; layer++)
    {
        if (layer != 0)
            A_cache.at(layer)->~Matrix();
        Z_cache.at(layer)->~Matrix();
        dW_cache.at(layer)->~Matrix();
        db_cache.at(layer)->~Matrix();
    }
    A_cache.clear();
    Z_cache.clear();
    dW_cache.clear();
    db_cache.clear();
}

int main()
{
    cout << "0. Initialize parameters" << endl;
    static double layer_dims[] = {784, 255, 10};
    len_layer = (*(&layer_dims + 1) - layer_dims) - 1;
    learning_rate = 0.001;
    epochs = 10;
    batchSize = 32;
    VERBOSE = true;

    Dataset train = Dataset(
        "../data/fashion_mnist_train_vectors.csv",
        "../data/fashion_mnist_train_labels.csv",
        3000, // size of train dataset
        VERBOSE);

    Dataset test = Dataset(
        "../data/fashion_mnist_test_vectors.csv",
        "../data/fashion_mnist_test_labels.csv",
        1000, // size of test dataset
        VERBOSE);

    vector<Matrix *> xBatch;
    vector<Matrix *> yBatch;
    int batchCount = initializeMiniBatch(xBatch, yBatch, &train, batchSize);

    cout << "Created " << batchCount << " batches\n";

    cout << "1. INITIALIZE PARAMETERS" << endl;
    vector<Matrix *> weights = initializeRandomWeights(layer_dims);
    vector<Matrix *> bias = initializeBias(layer_dims);

    cout << "NetworkLayers: ";
    for (int layer=0; layer<=len_layer; layer++) cout << layer_dims[layer] << ", ";
    cout << "\nBatchSize: " << batchSize << " Epochs: " << epochs << " LearningRate: " << learning_rate << endl; 

    cout << "2. LOOP " << endl;
    auto start = high_resolution_clock::now();
    for (int iteration = 0; iteration < epochs; iteration++)
    {
        for (int batch = 0; batch < batchCount; batch++)
        {
            // cout << " 2.1 FORWARD PROPAGATION" << endl;
            AL = forwardPropagation(xBatch.at(batch), weights, bias);

            // cout << " 2.2 COMPUTE COST" << endl;
            double loss = computeCostCrossEntropy(AL, yBatch.at(batch));

            // cout << " 2.3 BACKWARD PROPAGATION" << endl;
            backwardPropagation(AL, yBatch.at(batch), weights);

            // cout << " 2.4 UPDATE PARAMETERS" << endl;
            updateParametersGradientDescend(bias, weights);

            if ((batch == 0))
            {
                accuracy = predict(&test, weights, bias, "accuracy", false);
                cout << "[" << iteration << "] epoch ["<< batch << "] batch LOSS: " << loss << " ACC: " << accuracy << endl;
            }

            // cout << " 2.5 FREE CACHE" << endl;
            freeCache();
            AL->~Matrix();
        }
    }
    auto stop = high_resolution_clock::now();
    freeMatrixVector(xBatch, yBatch, batchCount);

    if (VERBOSE)
        cout << "Time for training " << duration_cast<milliseconds>(stop - start).count() << " milliseconds" << endl;

    cout << "3 PREDICT" << endl;
    {
        cout << "3.1 PREDICTION TRAIN" << endl;
        accuracy = predict(&train, weights, bias, "accuracy", VERBOSE);
        cout << "TRAIN [" << train.getY()->getColumns() << "] acc: " << accuracy << endl;

        cout << "3.2 PREDICTION TEST" << endl;
        accuracy = predict(&test, weights, bias, "accuracy", VERBOSE);
        cout << "TEST [" << test.getY()->getColumns() << "] acc: " << accuracy << endl;
    }

    if (VERBOSE)
        cout << "4. SAVE PARAMETERS" << endl;
    saveParameters(weights, bias);

    if (VERBOSE)
        cout << "5. FREE BIAS && WEIGHTS VECTOR" << endl;
    freeMatrixVector(weights, bias, len_layer);
    return 0;
}