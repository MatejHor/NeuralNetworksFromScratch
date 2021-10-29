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
static int num_iterations = 2;
// static int num_iterations = 1;
bool TEST = 0;

Matrix **A_cache = new Matrix *[len_layer];
Matrix **Z_cache = new Matrix *[len_layer];

Matrix **dA_cache = new Matrix *[len_layer];
Matrix **dW_cache = new Matrix *[len_layer];
Matrix **db_cache = new Matrix *[len_layer];

Matrix *initializeBias()
{
    Matrix *bias = new Matrix[len_layer + 1];
    for (int layer = 0; layer < len_layer; layer++)
    {
        bias[layer] = *(new Matrix(layer_dims[layer + 1], 1, 0.0));
        cout << "Initialize bias (rows=" << layer_dims[layer + 1] << ", columns=" << 1 << ", &=" << (&bias[layer]) << ")" << endl;
    }
    return bias;
}

Matrix *initializeWeights()
{
    Matrix *weights = new Matrix[len_layer + 1];
    for (int layer = 0; layer < len_layer; layer++)
    {
        weights[layer] = *(new Matrix(layer_dims[layer + 1], layer_dims[layer], 0.01));
        cout << "Initialize weights (rows=" << weights[layer].getRows() << ", columns=" << weights[layer].getColumns() << ", &=" << (&weights[layer]) << ")" << endl;
    }
    return weights;
}

Matrix *forwardPropagation(Matrix *X, Matrix *weights, Matrix *bias)
{
    Matrix *A_prev = X;
    Matrix *Z = NULL;
    Matrix *dot_W_A_prev;
    for (int hidden_layer = 0; hidden_layer < (len_layer - 1); hidden_layer++)
    {
        dot_W_A_prev = dot(&weights[hidden_layer], A_prev);
        Z = sumVector(dot_W_A_prev, &bias[hidden_layer]);

        // Destruct computed values
        if (A_cache[hidden_layer] != NULL && hidden_layer != 0){
            // cout << " --------------- in if --------------- " << endl;
            A_cache[hidden_layer]->~Matrix();
        }

        A_cache[hidden_layer] = A_prev;
        A_prev = reLu(Z);

        // Destruct computed values
        dot_W_A_prev->~Matrix();
        Z->~Matrix();
    }
    dot_W_A_prev = dot(&weights[(len_layer - 1)], A_prev);
    Z = sumVector(dot_W_A_prev, &bias[len_layer - 1]);

    // Destruct computed values
    if (A_cache[len_layer - 1] != NULL)
        A_cache[len_layer - 1]->~Matrix();

    A_cache[len_layer - 1] = A_prev;
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
    Matrix *logSubtrack_AL = log(subtrack(1, AL));
    Matrix *multiplySubtrack_Y_logSubtrack_AL = multiply(subtrack_Y, logSubtrack_AL);

    Matrix *sumMatrix_ = sum(multiply_Y_log_AL, multiplySubtrack_Y_logSubtrack_AL);
    double sumMatrixVal = sumMatrix(sumMatrix_);

    // Destruct computed values
    log_AL->~Matrix();
    multiply_Y_log_AL->~Matrix();
    subtrack_Y->~Matrix();
    logSubtrack_AL->~Matrix();
    multiplySubtrack_Y_logSubtrack_AL->~Matrix();
    sumMatrix_->~Matrix();
    return (-1.0 / m) * sumMatrixVal;
}

void backwardPropagation(Matrix *AL, Matrix *Y, Matrix *weights)
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

    Matrix *dot_dZ_AprevT = dot(dZ, A_cache[len_layer - 1]->T());
    Matrix *dW = multiply(dot_dZ_AprevT, (1.0 / m));
    Matrix *sumDimension_dZ = sumDimension(dZ);
    Matrix *db = multiply(sumDimension_dZ, (1.0 / m));
    Matrix *dA_prev = dot(weights[len_layer - 1].T(), dZ);

    // Destruct previous values
    if (dA_cache[len_layer - 1] != NULL)
    {
        dA_cache[len_layer - 1]->~Matrix();
        dW_cache[len_layer - 1]->~Matrix();
        db_cache[len_layer - 1]->~Matrix();
    }

    dA_cache[len_layer - 1] = dA_prev;
    dW_cache[len_layer - 1] = dW;
    db_cache[len_layer - 1] = db;

    // Destruct computed values
    dot_dZ_AprevT->~Matrix();
    sumDimension_dZ->~Matrix();
    dZ->~Matrix();

    // cout << "[+] LOOPING [+]" << endl;
    for (int hidden_layer = (len_layer - 2); hidden_layer >= 0; hidden_layer--)
    {
        // cout << "Hidden layer " << hidden_layer << endl;
        dZ = reLuDerivation(dA_prev);

        dot_dZ_AprevT = dot(dZ, A_cache[hidden_layer]->T());

        dW = multiply(dot_dZ_AprevT, (1.0 / m));
        sumDimension_dZ = sumDimension(dZ);
        db = multiply(sumDimension_dZ, (1.0 / m));
        dA_prev = dot(weights[hidden_layer].T(), dZ);

        // Destruct previous values
        if (dA_cache[hidden_layer] != NULL)
        {
            dA_cache[hidden_layer]->~Matrix();
            dW_cache[hidden_layer]->~Matrix();
            db_cache[hidden_layer]->~Matrix();
        }

        dA_cache[hidden_layer] = dA_prev;
        dW_cache[hidden_layer] = dW;
        db_cache[hidden_layer] = db;

        // Destruct computed values
        dot_dZ_AprevT->~Matrix();
        sumDimension_dZ->~Matrix();
        dZ->~Matrix();
    }
}

Matrix *updateWeights(Matrix *weights)
{
    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        Matrix *multiplydWLearningRate = multiply(dW_cache[hidden_layer], learning_rate);
        Matrix *previous_weights = &weights[hidden_layer];
        weights[hidden_layer] = *subtrack(&weights[hidden_layer], multiplydWLearningRate);

        // Destruct computed values
        // previous_weights->~Matrix();
        multiplydWLearningRate->~Matrix();
    }
    return weights;
}

Matrix *updateBias(Matrix *bias)
{
    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        Matrix *multiplydBLearningRate = multiply(db_cache[hidden_layer], learning_rate);
        bias[hidden_layer] = *subtrack(&bias[hidden_layer], multiplydBLearningRate);

        // Destruct computed values
        multiplydBLearningRate->~Matrix();
    }
    return bias;
}

Matrix *predict(Matrix *X, Matrix *weights, Matrix *bias)
{
    return forwardPropagation(X, weights, bias);
}

double f1_mikro(Matrix *Y, Matrix *AL)
{
    double **y = Y->getMatrix();
    double **al = AL->getMatrix();

    double TP = 0;
    double TN = 0;
    double FP = 0;
    double FN = 0;

    for (int i = 0; i < 10; i++)
    {
        for (int row = 0; row < Y->getRows(); row++)
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

double accuracy(Matrix *Y, Matrix *AL)
{
    double **y = Y->getMatrix();
    double **al = AL->getMatrix();

    double TP = 0;

    for (int row = 0; row < Y->getRows(); row++)
    {
        (al[row][0] == y[row][0] && ++TP);
    }

    return TP / (Y->getRows());
}

void saveParameters(Matrix *weights, Matrix *bias)
{
    ofstream weights_file;
    ofstream bias_file;
    weights_file.open("weights.txt");
    bias_file.open("bias.txt");

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

void fillNull()
{
    for (int i = 0; i < len_layer; i++)
    {
        A_cache[i] = NULL;
        Z_cache[i] = NULL;

        dA_cache[i] = NULL;
        dW_cache[i] = NULL;
        db_cache[i] = NULL;
    }
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
    fillNull();
    Matrix *weights = initializeWeights();
    Matrix *bias = initializeBias();
    Matrix *AL;

    cout << "2. LOOP" << endl;
    for (int iteration = 0; iteration <= num_iterations; iteration++)
    {
        cout << "[+] --------------- " << iteration << " --------------- [+}" << endl;
        cout << "2.1 FORWARD PROPAGATION" << endl;
        AL = forwardPropagation(train.getX(), weights, bias);
        AL->print();

        cout << "2.2 COMPUTE COST" << endl;
        double cost = computeCostCrossEntropy(AL, train.getY());

        cout << "2.3 BACKWARD PROPAGATION" << endl;
        backwardPropagation(AL, train.getY(), weights);

        cout << "2.4 UPDATE PARAMETERS" << endl;
        bias = updateBias(bias);
        weights = updateWeights(weights);

        if (iteration % 50 == 0 || true)
        {
            cout << "Cost after iteration " << iteration << ": " << cost << endl;
        }

        AL->~Matrix();
    }
    Matrix *TrueLabel = squeeze(train.getY(), "category");
    Matrix *PrediLabel = squeeze(AL, "max");
    cout << "Train f1_mikro: " << f1_mikro(TrueLabel, PrediLabel) << endl;
    cout << "Train accuracy: " << accuracy(TrueLabel, PrediLabel) << endl;

    // cout << "3. PREDICTION" << endl;
    // Dataset test = Dataset(
    //     "../data/fashion_mnist_test_vectors.csv",
    //     "../data/fashion_mnist_test_labels.csv");

    // AL = predict(test.getX(), weights, bias);
    // TrueLabel = squeeze(test.getY(), "category");
    // PrediLabel = squeeze(AL, "max");
    // cout << "Test F1_Mikro: " << f1_mikro(TrueLabel, PrediLabel);

    cout << "4. SAVE PARAMETERS" << endl;
    saveParameters(weights, bias);

    cout << "5. FREE BIAS & WEIGHTS" << endl;
    for (int i = 0; i < len_layer; i++)
    {
        bias[i].~Matrix();
        weights[i].~Matrix();
    }
    return 0;
}