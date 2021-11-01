#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(double* layer_dims, int epochs, int batchSize, double learning_rate, bool verbose): layer_dims(layer_dims), epochs(epochs), batchSize(batchSize), learning_rate(learning_rate), VERBOSE(verbose)
{
    len_layer = (*(&layer_dims + 1) - layer_dims) - 1;
}

void NeuralNetwork::initializeRandomParameters()
{
    weights = initializeRandomWeights(layer_dims);
    bias = initializeBias(layer_dims);
}

void NeuralNetwork::fit(Dataset *train, Dataset *validation){
    initializeMiniBatch(xBatch, yBatch, train, batchSize);
    Matrix* AL = NULL;

    auto start = high_resolution_clock::now();
    for (int iteration = 0; iteration < epochs; iteration++)
    {
        for (int batch = 0; batch < batchCount; batch++)
        {
            if (AL != NULL)
                AL->~Matrix();
            // cout << "2.1 FORWARD PROPAGATION" << endl;
            AL = forwardPropagation(xBatch.at(batch), weights, bias);

            // cout << "2.2 COMPUTE COST" << endl;
            double loss = computeCostCrossEntropy(AL, yBatch.at(batch));

            // cout << "2.3 BACKWARD PROPAGATION" << endl;
            backwardPropagation(AL, yBatch.at(batch), weights);

            // cout << "2.4 UPDATE PARAMETERS" << endl;
            updateParametersGradientDescend(bias, weights);

            // if (iteration % 10 == 0 && (batch == batchCount - 1))
            if (true && (batch == batchCount - 1))
            // if (iteration % 50 == 0 && (batch == batchCount - 1))
            {
                accuracy = predict(validation, weights, bias, "accuracy", false);
                cout << "[" << iteration << "] epoch LOSS: " << loss << " ACC: " << accuracy << endl;
            }

            // cout << "2.5 FREE CACHE" << endl;
            freeCache();
        }
    }
    auto stop = high_resolution_clock::now();
    freeMatrixVector(xBatch, yBatch, batchCount);
    if (VERBOSE)
        cout << "Time for training " << duration_cast<milliseconds>(stop - start).count() << " milliseconds" << endl;
}

vector<Matrix *> NeuralNetwork::initializeBias(double layer_dims[])
{
    vector<Matrix *> bias;
    for (int layer = 0; layer < len_layer; layer++)
    {
        bias.push_back(new Matrix(layer_dims[layer + 1], 1, 0.0));
    }
    return bias;
}

vector<Matrix *> NeuralNetwork::initializeRandomWeights(double layer_dims[])
{
    vector<Matrix *> weights;
    for (int layer = 0; layer < len_layer; layer++)
    {
        weights.push_back(new Matrix(layer_dims[layer + 1], layer_dims[layer], 0.01));
    }
    return weights;
}

vector<Matrix *> NeuralNetwork::initializeHeWeights(double layer_dims[])
{
    vector<Matrix *> weights;
    for (int layer = 0; layer < len_layer; layer++)
    {
        double he = sqrt(2 / layer_dims[layer]);
        weights.push_back(new Matrix(layer_dims[layer + 1], layer_dims[layer], he));
    }
    return weights;
}

Matrix *NeuralNetwork::forwardPropagation(Matrix *X, vector<Matrix *> weights, vector<Matrix *> bias)
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

double NeuralNetwork::computeCostCrossEntropy(Matrix *AL, Matrix *Y)
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

void NeuralNetwork::backwardPropagation(Matrix *AL, Matrix *Y, vector<Matrix *> weights)
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

void NeuralNetwork::updateParametersGradientDescend(vector<Matrix *> &bias, vector<Matrix *> &weights)
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

int NeuralNetwork::initializeMiniBatch(vector<Matrix *> &X, vector<Matrix *> &Y, Dataset *data, int batchSize)
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

double NeuralNetwork::predict(Dataset *data, vector<Matrix *> weights, vector<Matrix *> bias, string measure, bool verbose)
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

void NeuralNetwork::freeMatrixVector(vector<Matrix *> &vector1, vector<Matrix *> &vector2, int count)
{
    for (int i = 0; i < count; i++)
    {
        vector1.at(i)->~Matrix();
        vector2.at(i)->~Matrix();
    }
    vector1.clear();
    vector2.clear();
}

void NeuralNetwork::freeCache()
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

void NeuralNetwork::saveParameters(string path)
{
    ofstream weights_file;
    ofstream bias_file;
    weights_file.open(path + "weights.txt");
    bias_file.open(path + "bias.txt");

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