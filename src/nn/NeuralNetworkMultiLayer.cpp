#include "NeuralNetworkMultiLayer.h"

NeuralNetworkMultiLayer::NeuralNetworkMultiLayer(vector<int> layer, int epochs, int batchSize, double learningRate, double beta) : epoch(epochs), batchSize(batchSize), learningRate(learningRate), beta(beta), layer(layer)
{
    cout << "Creating object NeuralNetworkMultiLayer\n";
    cout << "Layers: {";
    for (auto const &c : layer)
        cout << c << ", ";
    cout << "} Epoch: " << epoch << " batchSize: " << batchSize << " learningRate: " << learningRate << " beta: " << beta << endl;
}

void NeuralNetworkMultiLayer::initialize()
{
    for (int layer = 1; layer < (this->layer.size()); layer++)
    {
        this->params["W" + to_string(layer)] = new Matrix(this->layer.at(layer), this->layer.at(layer - 1), sqrt(1. / this->layer.at(layer - 1)));
        this->params["b" + to_string(layer)] = new Matrix(this->layer.at(layer), 1, sqrt(1. / this->layer.at(layer - 1)));

        this->params["V_dW" + to_string(layer)] = new Matrix(this->layer.at(layer), this->layer.at(layer - 1), 0.0);
        this->params["V_db" + to_string(layer)] = new Matrix(this->layer.at(layer), 1, 0.0);
    }
}

void NeuralNetworkMultiLayer::forwardPropagation(Matrix *xBatch)
{
    // auto start = high_resolution_clock::now();
    this->cache["A0"] = xBatch;
    for (int layer = 1; layer < (this->layer.size() - 1); layer++)
    {
        this->cache["Z" + to_string(layer)] = forwardDot(this->params["W" + to_string(layer)], this->cache["A" + to_string(layer - 1)], this->params["b" + to_string(layer)]);
        this->cache["A" + to_string(layer)] = sigmoid(this->cache["Z" + to_string(layer)]);
    }
    this->cache["Z" + to_string(this->layer.size() - 1)] = forwardDot(this->params["W" + to_string(this->layer.size() - 1)], this->cache["A" + to_string(this->layer.size() - 2)], this->params["b" + to_string(this->layer.size() - 1)]);
    this->cache["A" + to_string(this->layer.size() - 1)] = softmax(this->cache["Z" + to_string(this->layer.size() - 1)]);
    // auto stop = high_resolution_clock::now();
    // cout << "Forward: " << duration_cast<milliseconds>(stop - start).count() << " milliseconds" << endl;
}

double NeuralNetworkMultiLayer::costCrossEntropy(Matrix *AL, Matrix *yBatch)
{
    double sumMultiply_Y_logAL = crossEntropySum(yBatch, AL);
    return -(1.0 / yBatch->getColumns()) * sumMultiply_Y_logAL;
}

void NeuralNetworkMultiLayer::backPropagation(Matrix *xBatch, Matrix *yBatch, double m_batch)
{
    // auto start = high_resolution_clock::now();
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
    // auto stop = high_resolution_clock::now();
    // cout << "Backprop: " << duration_cast<milliseconds>(stop - start).count() << " milliseconds" << endl;
}

void NeuralNetworkMultiLayer::clearCache(bool clearGrads)
{
    for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
    {
        this->cache["Z" + to_string(layer)]->~Matrix();
        this->cache["A" + to_string(layer)]->~Matrix();
        if (clearGrads) {
            this->grads["dW" + to_string(layer)]->~Matrix();
            this->grads["db" + to_string(layer)]->~Matrix();
        }
    }
}

void NeuralNetworkMultiLayer::fit(Dataset *train, Dataset *test)
{
    vector<Matrix *> X;
    vector<Matrix *> Y;
    vector<int> batch_index;
    vector<double> costs;
    auto rng = default_random_engine{};
    double acc;
    int offset = 0;
    int length_data = train->getMaxRow();

    // CREATE BATCH FROM TRAIN
    while (length_data != 0)
    {
        if (length_data - this->batchSize < 0)
            this->batchSize = length_data;

        X.push_back(new Matrix(train->getX(), batchSize, offset));
        Y.push_back(new Matrix(train->getY(), batchSize, offset));

        length_data -= this->batchSize;
        offset += this->batchSize;
    }

    // FILL RANDOM INDEX ARRAY
    for (int i = 0; i <  X.size(); i++)
            batch_index.push_back(i);

    this->initialize(); 

    for (int epoch = 0; epoch < this->epoch; epoch++)
    {
        shuffle(batch_index.begin(), batch_index.end(), rng);

        auto start = high_resolution_clock::now();
        for (int batch = 0; batch < X.size(); batch++)
        {
            Matrix *xBatch = X.at(batch_index.at(batch));
            Matrix *yBatch = Y.at(batch_index.at(batch));

            int batchLength = xBatch->getColumns();

            this->forwardPropagation(xBatch);
            this->backPropagation(xBatch, yBatch, batchLength);

            for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
            {
                // ??? UPRAVA UPDATE PREMENIAME POVODNU MATICU ???
                // Matrix *V_dW_cache = this->params["V_dW" + to_string(layer)];
                // Matrix *V_db_cache = this->params["V_db" + to_string(layer)];
                this->params["V_dW" + to_string(layer)] = momentumSum(this->params["V_dW" + to_string(layer)], this->beta, this->grads["dW" + to_string(layer)], (1.0 - this->beta));
                this->params["V_db" + to_string(layer)] = momentumSum(this->params["V_db" + to_string(layer)], this->beta, this->grads["db" + to_string(layer)], (1.0 - this->beta));

                // V_db_cache->~Matrix();
                // V_dW_cache->~Matrix();
            }

            for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
            {
                // ??? UPRAVA UPDATE PREMENIAME POVODNU MATICU ???
                // Matrix *W_cache = this->params["W" + to_string(layer)];
                // Matrix *b_cache = this->params["b" + to_string(layer)];
                this->params["W" + to_string(layer)] = momentumUpdate(this->params["W" + to_string(layer)], this->params["V_dW" + to_string(layer)], this->learningRate);
                this->params["b" + to_string(layer)] = momentumUpdate(this->params["b" + to_string(layer)], this->params["V_db" + to_string(layer)], this->learningRate);

                // W_cache->~Matrix();
                // b_cache->~Matrix();
            }
            costs.push_back(costCrossEntropy(this->cache["A" + to_string(this->layer.size() - 1)], yBatch));
            
            this->clearCache(true);
        }
        auto stop = high_resolution_clock::now();
            
        double cost = std::accumulate(costs.begin(), costs.end(), 0.0) / costs.size();    
        cout << "Epoch [" << epoch << "] training cost: " << cost << " training time": << duration_cast<milliseconds>(stop - start).count() << " milliseconds" << endl;

        // double previous_acc = acc;
        // acc = this->transform(test);
        // cout << " Accuracy: " << acc << endl;
    }
}

double NeuralNetworkMultiLayer::transform(Dataset *test)
{
    this->forwardPropagation(test->getX());
    double acc = accuracy(this->cache["A" + to_string(this->layer.size() - 1)], test->getY());
    this->clearCache(false);
    return acc;
}