#include "NeuralNetworkMultiLayer.h"

NeuralNetworkMultiLayer::NeuralNetworkMultiLayer(vector<int> layer, int epochs, int batchSize, double learningRate, double beta) : epoch(epochs), batchSize(batchSize), learningRate(learningRate), beta(beta), layer(layer)
{
    cout << "Creating object NeuralNetworkMultiLayer\n";
    cout << "Layers: {"
    for (auto const &c : layer)
        cout << c << ', ';
    cout << "} Epoch: " << epoch << " batchSize: " << batchSize << " learningRate: " << learningRate << " beta: " << beta << endl;
}

void NeuralNetworkMultiLayer::initialize()
{
    for (int layer = 1; layer < (this->layer.size()); layer++)
    {
        this->params["W" + to_string(layer)] = new Matrix(this->layer.at(layer), this->layer.at(layer - 1), sqrt(1. / this->layer.at(layer - 1)));
        // this->params["b" + to_string(layer)] = new Matrix(this->layer.at(layer), 1, 0.0);
        this->params["b" + to_string(layer)] = new Matrix(this->layer.at(layer), 1, sqrt(1. / this->layer.at(layer - 1)));

        this->params["V_dW" + to_string(layer)] = new Matrix(this->layer.at(layer), this->layer.at(layer - 1), 0.0);
        this->params["V_db" + to_string(layer)] = new Matrix(this->layer.at(layer), 1, 0.0);
    }
}

void NeuralNetworkMultiLayer::forwardPropagation(Matrix *xBatch)
{
    this->cache["A0"] = xBatch;
    for (int layer = 1; layer < (this->layer.size() - 1); layer++)
    {
        this->cache["Z" + to_string(layer)] = forwardDot(this->params["W" + to_string(layer)], this->cache["A" + to_string(layer - 1)], this->params["b" + to_string(layer)]);
        this->cache["A" + to_string(layer)] = sigmoid(this->cache["Z" + to_string(layer)]);
    }
    this->cache["Z" + to_string(this->layer.size() - 1)] = forwardDot(this->params["W" + to_string(this->layer.size() - 1)], this->cache["A" + to_string(this->layer.size() - 2)], this->params["b" + to_string(this->layer.size() - 1)]);
    this->cache["A" + to_string(this->layer.size() - 1)] = softmax(this->cache["Z" + to_string(this->layer.size() - 1)]);
}

double NeuralNetworkMultiLayer::costCrossEntropy(Matrix *AL, Matrix *yBatch)
{
    double sumMultiply_Y_logAL = crossEntropySum(yBatch, AL);

    return -(1.0 / yBatch->getColumns()) * sumMultiply_Y_logAL;
}

void NeuralNetworkMultiLayer::backPropagation(Matrix *xBatch, Matrix *yBatch, double m_batch)
{
    Matrix *dZ = subtrack(this->cache["A" + to_string((this->layer.size() - 1))], yBatch);

    for (int layer = (this->layer.size() - 1); layer > 0; layer--)
    {
        this->grads["dW" + to_string(layer)] = backwardDotDW(dZ, this->cache["A" + to_string(layer - 1)], (1.0 / m_batch));
        this->grads["db" + to_string(layer)] = backwardSumDimension(dZ, (1.0 / m_batch));

        Matrix *dZ_cache = dZ;
        if (layer != 1)
            dZ = backwardDotDZSigmoid(this->params["W" + to_string(layer)], dZ, this->cache["Z" + to_string(layer - 1)]);
        dZ_cache->~Matrix();
    }

    this->cache["A0"] = NULL;
}

void NeuralNetworkMultiLayer::clearCache()
{
    for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
    {
        this->cache["Z" + to_string(layer)]->~Matrix();
        this->cache["A" + to_string(layer)]->~Matrix();
    }
}

void NeuralNetworkMultiLayer::fit(Dataset *train)
{
    vector<Matrix *> X;
    vector<Matrix *> Y;
    vector<int> arra;
    double acc;
    auto rng = default_random_engine{};
    int offset = 0;
    int length_data = train->getMaxRow();
    while (length_data != 0)
    {
        if (length_data - this->batchSize < 0)
            this->batchSize = length_data;
        for (int i = offset; i < batchSize + offset; i++)
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

            for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
            {
                Matrix *V_dW_cache = this->params["V_dW" + to_string(layer)];
                this->params["V_dW" + to_string(layer)] = momentumSum(this->params["V_dW" + to_string(layer)], this->beta, this->grads["dW" + to_string(layer)], (1.0 - this->beta));
                Matrix *V_db_cache = this->params["V_db" + to_string(layer)];
                this->params["V_db" + to_string(layer)] = momentumSum(this->params["V_db" + to_string(layer)], this->beta, this->grads["db" + to_string(layer)], (1.0 - this->beta));

                V_db_cache->~Matrix();
                V_dW_cache->~Matrix();
            }

            for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
            {
                Matrix *W_cache = this->params["W" + to_string(layer)];
                this->params["W" + to_string(layer)] = momentumUpdate(W_cache, this->params["V_dW" + to_string(layer)], this->learningRate);
                Matrix *b_cache = this->params["b" + to_string(layer)];
                this->params["b" + to_string(layer)] = momentumUpdate(b_cache, this->params["V_db" + to_string(layer)], this->learningRate);

                W_cache->~Matrix();
                b_cache->~Matrix();

                this->grads["dW" + to_string(layer)]->~Matrix();
                this->grads["db" + to_string(layer)]->~Matrix();
            }
            this->clearCache();
        }
        this->forwardPropagation(train->getX());
        double cost = this->costCrossEntropy(this->cache["A" + to_string(this->layer.size() - 1)], train->getY());
        this->clearCache();

        double previous_acc = acc;
        acc = this->transform(train);
        // if (previous_acc - acc <= 0.01)
        //     this->learningRate /= 10;
        cout << "Epoch [" << epoch << "] training cost: " << cost << " Accuracy: " << acc << endl;
    }
}

double NeuralNetworkMultiLayer::transform(Dataset *test)
{
    this->forwardPropagation(test->getX());
    double acc = accuracy(this->cache["A" + to_string(this->layer.size() - 1)], test->getY());
    this->clearCache();
    return acc;
}