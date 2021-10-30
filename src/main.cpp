#include "./matrix/Matrix.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./dataset/Dataset.cpp"
#include <stdlib.h> // RAND

#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>

static double layer_dims[] = {784, 255, 225, 200, 10};
static int len_layer = 4;
static double learning_rate = 0.0075;
// static int num_iterations = 500;
static int num_iterations = 1;
bool TEST = 0;

vector<Matrix> A_cache;
vector<Matrix> dA_cache;
vector<Matrix> dW_cache;
vector<Matrix> db_cache;

vector<Matrix> initializeBias()
{
    vector<Matrix> bias;
    for (int layer = 0; layer < len_layer; layer++)
    {
       bias.push_back(Matrix(layer_dims[layer + 1], 1, 0.0));
    }
    return bias;
}

vector<Matrix> initializeWeights()
{
    vector<Matrix> weights;
    for (int layer = 0; layer < len_layer; layer++)
    {
        weights.push_back(Matrix(layer_dims[layer + 1], layer_dims[layer], 0.01));
    }
    return weights;
}

Matrix *forwardPropagation(Matrix *X, vector<Matrix> weights, vector<Matrix> bias)
{
    Matrix *A_prev = X;
    Matrix *Z = NULL;
    Matrix *dot_W_A_prev;

    for (int hidden_layer = 0; hidden_layer < (len_layer - 1); hidden_layer++)
    {
        dot_W_A_prev = dot(&weights.at(hidden_layer), A_prev);
        Z = sumVector(dot_W_A_prev, &bias.at(hidden_layer));

        A_cache.at(hidden_layer) = *A_prev;
        A_prev = reLu(Z);

        // Destruct computed values
        dot_W_A_prev->~Matrix();
        Z->~Matrix();
    }
    dot_W_A_prev = dot(&weights.at(len_layer - 1), A_prev);
    Z = sumVector(dot_W_A_prev, &bias.at(len_layer - 1));

    A_cache.at(len_layer - 1) = *A_prev;
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

void backwardPropagation(Matrix *AL, Matrix *Y, vector<Matrix> weights)
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

    Matrix *trans_A_cache = (A_cache.at(len_layer - 1)).T();
    Matrix *dot_dZ_AprevT = dot(dZ, trans_A_cache);
    Matrix *dW = multiply(dot_dZ_AprevT, (1.0 / m));
    Matrix *sumDimension_dZ = sumDimension(dZ);
    Matrix *db = multiply(sumDimension_dZ, (1.0 / m));
    Matrix *trans_weight = (weights.at(len_layer - 1)).T();
    Matrix *dA_prev = dot(trans_weight, dZ);

    dA_cache.at(len_layer - 1) = *dA_prev;
    dW_cache.at(len_layer - 1) = *dW;
    db_cache.at(len_layer - 1) = *db;

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

        trans_A_cache = (A_cache.at(hidden_layer)).T();
        dot_dZ_AprevT = dot(dZ, trans_A_cache);
        dW = multiply(dot_dZ_AprevT, (1.0 / m));
        sumDimension_dZ = sumDimension(dZ);
        db = multiply(sumDimension_dZ, (1.0 / m));
        trans_weight = (weights.at(hidden_layer)).T();
        dA_prev = dot(trans_weight, dZ);

        dA_cache.at(hidden_layer) = *dA_prev;
        dW_cache.at(hidden_layer) = *dW;
        db_cache.at(hidden_layer) = *db;

        // Destruct computed values
        trans_weight->~Matrix();
        trans_A_cache->~Matrix();
        dot_dZ_AprevT->~Matrix();
        sumDimension_dZ->~Matrix();
        dZ->~Matrix();
    }
}

vector<Matrix> updateWeights(vector<Matrix> weights)
{
    vector<Matrix> new_weights; // TODO: matrix -> vector
    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        Matrix *multiplydWLearningRate = multiply(&dW_cache.at(hidden_layer), learning_rate);
        new_weights.at(hidden_layer) = *subtrack((&weights.at(hidden_layer)), multiplydWLearningRate);

        // Destruct computed values
        multiplydWLearningRate->~Matrix();
    }
    weights.clear();
    return new_weights;
}

vector<Matrix> updateBias(vector<Matrix> bias)
{
    vector<Matrix> new_bias;
    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        Matrix *multiplydBLearningRate = multiply(&db_cache.at(hidden_layer), learning_rate);
        new_bias.at(hidden_layer) = *subtrack(&bias.at(hidden_layer), multiplydBLearningRate);

        // Destruct computed values
        multiplydBLearningRate->~Matrix();
    }
    bias.clear();
    return new_bias;
}

void freeCache(){
    A_cache.clear();
    dA_cache.clear();
    dW_cache.clear();
    db_cache.clear();
}

Matrix *predict(Matrix *X, vector<Matrix> weights, vector<Matrix> bias)
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

void saveParameters(vector<Matrix> weights, vector<Matrix> bias)
{
    ofstream weights_file;
    ofstream bias_file;
    weights_file.open("weights.txt");
    bias_file.open("bias.txt");

    for (int hidden_layer = 0; hidden_layer < len_layer; hidden_layer++)
    {
        string weights_value = "";
        string bias_value = "";
        double **matrix_weights = (weights.at(hidden_layer)).getMatrix();
        double **matrix_bias = (bias.at(hidden_layer)).getMatrix();
        int rows = (weights.at(hidden_layer)).getRows();
        int columns = (weights.at(hidden_layer)).getColumns();

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
    vector<Matrix> weights = initializeWeights();
    vector<Matrix> bias = initializeBias();
    Matrix *AL = NULL;

    cout << "2. LOOP" << endl;
    for (int iteration = 0; iteration < num_iterations; iteration++)
    {
        if (AL != NULL) AL->~Matrix();
        cout << "[+] --------------- " << iteration << " --------------- [+}" << endl;
        cout << "2.1 FORWARD PROPAGATION" << endl;
        AL = forwardPropagation(train.getX(), weights, bias);

        cout << "2.2 COMPUTE COST" << endl;
        double cost = computeCostCrossEntropy(AL, train.getY());

        cout << "2.3 BACKWARD PROPAGATION" << endl;
        backwardPropagation(AL, train.getY(), weights);

        cout << "2.4 UPDATE PARAMETERS" << endl;
        weights = updateWeights(weights);
        bias = updateBias(bias);

        if (iteration % 50 == 0 || true)
        {
            cout << "Cost after iteration " << iteration << ": " << cost << endl;
        }

        cout << "2.5 FREE CACHE" << endl;
        freeCache();
    }
    // Matrix *TrueLabel = squeeze(train.getY(), "category");
    // Matrix *PrediLabel = squeeze(AL, "max");
    // cout << "Train f1_mikro: " << f1_mikro(TrueLabel, PrediLabel) << endl;
    // cout << "Train accuracy: " << accuracy(TrueLabel, PrediLabel) << endl;

    // cout << "3. PREDICTION" << endl;
    // Dataset test = Dataset(
    //     "../data/fashion_mnist_test_vectors.csv",
    //     "../data/fashion_mnist_test_labels.csv");

    // AL = predict(test.getX(), weights, bias);
    // TrueLabel = squeeze(test.getY(), "category");
    // PrediLabel = squeeze(AL, "max");
    // cout << "Test F1_Mikro: " << f1_mikro(TrueLabel, PrediLabel);

    // cout << "4. SAVE PARAMETERS" << endl;
    // saveParameters(weights, bias);

    cout << "5. FREE BIAS & WEIGHTS VECTOR" << endl;
    bias.clear();
    weights.clear();
    return 0;
}