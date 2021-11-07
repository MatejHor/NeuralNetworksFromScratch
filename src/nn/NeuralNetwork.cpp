#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int epochs, int batchSize, double learningRate, double beta) : epoch(epochs), batchSize(batchSize), learningRate(learningRate), beta(beta)
{
    cout << "Creating object NeuralNetwork";
    cout << " Epoch: " << epoch << " batchSize: " << batchSize << " learningRate: " << learningRate << " beta: " << beta << endl;
}

void NeuralNetwork::initialize()
{
    this->params["W1"] = new Matrix(256, 784, sqrt(1. / 784));
    this->params["b1"] = new Matrix(256, 1, 0.0);
    this->params["W2"] = new Matrix(10, 256, sqrt(1. / 256));
    this->params["b2"] = new Matrix(10, 1, 0.0);

    this->params["V_dW1"] = new Matrix(256, 784, 0.0);
    this->params["V_db1"] = new Matrix(256, 1, 0.0);
    this->params["V_dW2"] = new Matrix(10, 256, 0.0);
    this->params["V_db2"] = new Matrix(10, 1, 0.0);
}

void NeuralNetwork::forwardPropagation(Matrix *xBatch)
{
    {
        // Matrix *dot_W1_X = dot(this->params["W1"], xBatch);
        // this->cache["Z1"] = sumVector(dot_W1_X, this->params["b1"]);
        // this->cache["A1"] = sigmoid(this->cache["Z1"]);

        // Matrix *dot_W2_A1 = dot(this->params["W2"], this->cache["A1"]);
        // this->cache["Z2"] = sumVector(dot_W2_A1, this->params["b2"]);
        // this->cache["A2"] = softmax(this->cache["Z2"]);
        // {
        //     dot_W1_X->~Matrix();
        //     dot_W2_A1->~Matrix();
        // }
    }

    this->cache["Z1"] = forwardDot(this->params["W1"], xBatch, this->params["b1"]);
    this->cache["A1"] = sigmoid(this->cache["Z1"]);

    this->cache["Z2"] = forwardDot(this->params["W2"], this->cache["A1"], this->params["b2"]);
    this->cache["A2"] = softmax(this->cache["Z2"]);
}

double NeuralNetwork::costCrossEntropy(Matrix *AL, Matrix *yBatch)
{
    {
        // Matrix *logAl = log(AL);
        // Matrix *multiply_Y_logAL = multiply(yBatch, logAl);
        // double sumMultiply_Y_logAL = sumMatrix(multiply_Y_logAL);
        // logAl->~Matrix();
        // multiply_Y_logAL->~Matrix();
    }

    double sumMultiply_Y_logAL = crossEntropySum(yBatch, AL);

    return -(1.0 / yBatch->getColumns()) * sumMultiply_Y_logAL;
}

void NeuralNetwork::backPropagation(Matrix *xBatch, Matrix *yBatch, double m_batch)
{
    { // Matrix *dZ2 = subtrack(this->cache["A2"], yBatch);
        // Matrix *A1_T = this->cache["A1"]->T();
        // Matrix *dot_dZ2_A1T = dot(dZ2, A1_T);
        // Matrix *sumDimension_dZ2 = sumDimension(dZ2);
        // this->grads["dW2"] = multiply(dot_dZ2_A1T, (1.0 / m_batch));
        // this->grads["db2"] = multiply(sumDimension_dZ2, (1.0 / m_batch));

        // Matrix *W2_T = this->params["W2"]->T();
        // Matrix *sigmoidDerivative_Z1 = sigmoidDerivative(this->cache["Z1"]);
        // Matrix *dA1 = dot(W2_T, dZ2);
        // Matrix *dZ1 = multiply(dA1, sigmoidDerivative_Z1);
        // Matrix *X_T = xBatch->T();
        // Matrix *dot_dZ1_XT = dot(dZ1, X_T);
        // Matrix *sumDimension_dZ1 = sumDimension(dZ1);
        // this->grads["dW1"] = multiply(dot_dZ1_XT, (1.0 / m_batch));
        // this->grads["db1"] = multiply(sumDimension_dZ1, (1.0 / m_batch));
        {
            // dZ2->~Matrix();
            // sigmoidDerivative_Z1->~Matrix();
            // dZ1->~Matrix();
            // sumDimension_dZ2->~Matrix();
            // sumDimension_dZ1->~Matrix();
            // A1_T->~Matrix();
            // dot_dZ2_A1T->~Matrix();
            // W2_T->~Matrix();
            // X_T->~Matrix();
            // dot_dZ1_XT->~Matrix();
        }
    }

    Matrix *dZ2 = subtrack(this->cache["A2"], yBatch);
    this->grads["dW2"] = backwardDotDW(dZ2, this->cache["A1"], (1.0 / m_batch));
    this->grads["db2"] = backwardSumDimension(dZ2, (1.0 / m_batch));

    Matrix *dZ1 = backwardDotDZSigmoid(this->params["W2"], dZ2, this->cache["Z1"]);
    this->grads["dW1"] = backwardDotDW(dZ1, xBatch, (1.0 / m_batch));
    this->grads["db1"] = backwardSumDimension(dZ1, (1.0 / m_batch));

    {
        dZ2->~Matrix();
        dZ1->~Matrix();
    }
}

void NeuralNetwork::clearCache()
{
    this->cache["Z1"]->~Matrix();
    this->cache["A1"]->~Matrix();
    this->cache["Z2"]->~Matrix();
    this->cache["A2"]->~Matrix();
}

void NeuralNetwork::fit(Dataset *train)
{
    vector<Matrix *> X;
    vector<Matrix *> Y;
    vector<int> arra;
    double acc;
    auto rng = default_random_engine {};
    int offset = 0;
    int length_data = train->getMaxRow();
    while (length_data != 0)
    {
        if (length_data - this->batchSize < 0)
            this->batchSize = length_data;
        for(int i = offset; i < batchSize + offset; i++)
            arra.push_back(i);

        shuffle(arra.begin(), arra.end(), rng);

        X.push_back(new Matrix(train->getX(), batchSize, arra));
        Y.push_back(new Matrix(train->getY(), batchSize, arra));

        length_data -= this->batchSize;
        offset += this->batchSize;
        arra.clear();
    }

    this->initialize();

    for (int epoch = 0; epoch < this->epoch; epoch++)
    {
        for (int batch = 0; batch < X.size(); batch++)
        {
            Matrix *xBatch = X.at(batch);
            Matrix *yBatch = Y.at(batch);

            int batchLength = xBatch->getColumns();

            this->forwardPropagation(xBatch);
            this->backPropagation(xBatch, yBatch, batchLength);

            // MOMENTUM UPDATE
            {
                // Matrix *V_dW1 = this->params["V_dW1"];
                // Matrix *multiply_VdW1 = multiply(this->params["V_dW1"], this->beta);
                // Matrix *multiply_dW1 = multiply(this->grads["dW1"], (1.0 - this->beta));
                // this->params["V_dW1"] = sum(multiply_VdW1, multiply_dW1);

                // Matrix *V_db1 = this->params["V_db1"];
                // Matrix *multiply_Vdb1 = multiply(this->params["V_db1"], this->beta);
                // Matrix *multiply_db1 = multiply(this->grads["db1"], (1.0 - this->beta));
                // this->params["V_db1"] = sum(multiply_Vdb1, multiply_db1);

                // Matrix *V_dW2 = this->params["V_dW2"];
                // Matrix *multiply_VdW2 = multiply(this->params["V_dW2"], this->beta);
                // Matrix *multiply_dW2 = multiply(this->grads["dW2"], (1.0 - this->beta));
                // this->params["V_dW2"] = sum(multiply_VdW2, multiply_dW2);

                // Matrix *V_db2 = this->params["V_db2"];
                // Matrix *multiply_Vdb2 = multiply(this->params["V_db2"], this->beta);
                // Matrix *multiply_db2 = multiply(this->grads["db2"], (1.0 - this->beta));
                // this->params["V_db2"] = sum(multiply_Vdb2, multiply_db2);
                {
                    // multiply_VdW1->~Matrix();
                    // multiply_dW1->~Matrix();
                    // V_dW1->~Matrix();
                    // V_db1->~Matrix();
                    // V_dW2->~Matrix();
                    // V_db2->~Matrix();
                    // multiply_Vdb1->~Matrix();
                    // multiply_db1->~Matrix();
                    // multiply_VdW2->~Matrix();
                    // multiply_dW2->~Matrix();
                    // multiply_Vdb2->~Matrix();
                    // multiply_db2->~Matrix();
                }
            }

            Matrix *V_dW1 = this->params["V_dW1"];
            this->params["V_dW1"] = momentumSum(this->params["V_dW1"], this->beta, this->grads["dW1"], (1.0 - this->beta));
            Matrix *V_db1 = this->params["V_db1"];
            this->params["V_db1"] = momentumSum(this->params["V_db1"], this->beta, this->grads["db1"], (1.0 - this->beta));
            Matrix *V_dW2 = this->params["V_dW2"];
            this->params["V_dW2"] = momentumSum(this->params["V_dW2"], this->beta, this->grads["dW2"], (1.0 - this->beta));
            Matrix *V_db2 = this->params["V_db2"];
            this->params["V_db2"] = momentumSum(this->params["V_db2"], this->beta, this->grads["db2"], (1.0 - this->beta));

            // UPDATE PARAMS
            {
                // Matrix *W1 = this->params["W1"];
                // Matrix *multiply_LR_V_dW1 = multiply(this->params["V_dW1"], this->learningRate);
                // this->params["W1"] = subtrack(W1, multiply_LR_V_dW1);

                // Matrix *b1 = this->params["b1"];
                // Matrix *multiply_LR_V_db1 = multiply(this->params["V_db1"], this->learningRate);
                // this->params["b1"] = subtrack(b1, multiply_LR_V_db1);

                // Matrix *W2 = this->params["W2"];
                // Matrix *multiply_LR_V_dW2 = multiply(this->params["V_dW2"], this->learningRate);
                // this->params["W2"] = subtrack(W2, multiply_LR_V_dW2);

                // Matrix *b2 = this->params["b2"];
                // Matrix *multiply_LR_V_db2 = multiply(this->params["V_db2"], this->learningRate);
                // this->params["b2"] = subtrack(b2, multiply_LR_V_db2);

                {
                    // multiply_LR_V_dW1->~Matrix();
                    // multiply_LR_V_db1->~Matrix();
                    // W1->~Matrix();
                    // b1->~Matrix();
                    // W2->~Matrix();
                    // b2->~Matrix();
                    // multiply_LR_V_dW2->~Matrix();
                    // multiply_LR_V_db2->~Matrix();
                }
            }

            Matrix *W1 = this->params["W1"];
            this->params["W1"] = momentumUpdate(W1, this->params["V_dW1"], this->learningRate);
            Matrix *b1 = this->params["b1"];
            this->params["b1"] = momentumUpdate(b1, this->params["V_db1"], this->learningRate);
            Matrix *W2 = this->params["W2"];
            this->params["W2"] = momentumUpdate(W2, this->params["V_dW2"], this->learningRate);
            Matrix *b2 = this->params["b2"];
            this->params["b2"] = momentumUpdate(b2, this->params["V_db2"], this->learningRate);

            {
                W1->~Matrix();
                b1->~Matrix();
                W2->~Matrix();
                b2->~Matrix();

                V_dW1->~Matrix();
                V_db1->~Matrix();
                V_dW2->~Matrix();
                V_db2->~Matrix();

                this->grads["dW1"]->~Matrix();
                this->grads["db1"]->~Matrix();
                this->grads["dW2"]->~Matrix();
                this->grads["db2"]->~Matrix();
                this->clearCache();
            }
        }
        this->forwardPropagation(train->getX());
        double cost = this->costCrossEntropy(this->cache["A2"], train->getY());
        this->clearCache();

        double previous_acc = acc;
        acc = this->transform(train);
        // if (previous_acc - acc <= 0.01) 
        //     this->learningRate /= 10;
        cout << "Epoch [" << epoch << "] training cost: " << cost << " Accuracy: " << acc << endl;
    }
}

double NeuralNetwork::transform(Dataset *test)
{
    this->forwardPropagation(test->getX());
    double acc = accuracy(this->cache["A2"], test->getY());
    this->clearCache();
    return acc;
}