#include "NeuralNetworkMultiLayer.h"

/**
 * @brief Construct a new Neural Network Multi Layer:: Neural Network Multi Layer object
 * 
 * @param layer hierarchy of network, number of neurons in layer
 * @param epochs number of epoch which will be network trained
 * @param batchSize size of batch for data
 * @param learningRate learning rate for update weight and bias
 * @param beta beta parameter for momentum
 */
NeuralNetworkMultiLayer::NeuralNetworkMultiLayer(vector<int> layer, int epochs, int batchSize, double learningRate, double beta) : epoch(epochs), batchSize(batchSize), learningRate(learningRate), beta(beta), layer(layer)
{
    cout << "Creating object NeuralNetworkMultiLayer\n";
    cout << "Layers: {";
    for (auto const &c : layer)
        cout << c << ", ";
    cout << "} Epoch: " << epoch << " batchSize: " << batchSize << " learningRate: " << learningRate << " beta: " << beta << endl;
}

/**
 * @brief Create matrix for weights, bias and momentum. 
 * Weiths and bias initialize with random value multiply by sqrt(1 / previous layer), Momentum initialize with 0.
 */
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

/**
 * @brief Forward data @param xBatch to get prediction. First A (A0) is xBatch data. 
 * Use sigmoid on hidden layer and on output layer use softmax.
 * To compute Z_i use formula dot(Weight_i, A_i-1) * Bias_i after that applied sigmoid (on output softmax).
 *  
 * @param xBatch data forwarded over the network
 */
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

/**
 * @brief Compute crossEntropy loss function from @param AL and @param yBatch.
 * Formula (-1/n) * sum(Y - log(AL))
 * 
 * @param AL matrix of prediction
 * @param yBatch matrix of true result 
 * @return computed cross entropy value (double)
 */
double NeuralNetworkMultiLayer::costCrossEntropy(Matrix *AL, Matrix *yBatch)
{
    double sumMultiply_Y_logAL = crossEntropySum(yBatch, AL);
    return -(1.0 / yBatch->getColumns()) * sumMultiply_Y_logAL;
}

/**
 * @brief compute derivate of weights and bias.
 * To get dW use formula dot(dZ_i, A_i-1) * (1/m_batch).
 * To get dB use formula sumDimension(dZ_i) * (1/m_batch).
 * To get dZ use formual dot(W_i, dZ_previous) * derivateSigmoid(Z_i-1) and for first dZ (A_last - Y)
 * 
 * @param yBatch matrix of true labels, used for first derivative
 * @param m_batch length of batch, used to get better derivative of dW and dB
 */
void NeuralNetworkMultiLayer::backPropagation(Matrix *yBatch, double m_batch)
{
    // First computation of dZ
    Matrix *dZ = subtrack(this->cache["A" + to_string((this->layer.size() - 1))], yBatch);

    for (int layer = (this->layer.size() - 1); layer > 0; layer--)
    {
        // Compute dW
        this->grads["dW" + to_string(layer)] = backwardDotDW(dZ, this->cache["A" + to_string(layer - 1)], (1.0 / m_batch));
        // Compute db
        this->grads["db" + to_string(layer)] = backwardSumDimension(dZ, (1.0 / m_batch));

        // Save previous dZ to dealocate it
        Matrix *dZ_cache = dZ;
        // Not needed to compute derivate for first layer
        if (layer != 1)
            dZ = backwardDotDZSigmoid(this->params["W" + to_string(layer)], dZ, this->cache["Z" + to_string(layer - 1)]);
        dZ_cache->~Matrix();
    }

    // Remove xBatch from cache
    this->cache["A0"] = NULL;
}

/**
 * @brief Dealocate Matrices in cache and if requeired ( @param clearGrads ) also dealocate Matrix in grads
 * 
 * @param clearGrads True if is requeired to dealocate grads data (not needed in transform)
 */
void NeuralNetworkMultiLayer::clearCache(bool clearGrads)
{
    for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
    {
        this->cache["Z" + to_string(layer)]->~Matrix();
        this->cache["A" + to_string(layer)]->~Matrix();
        if (clearGrads)
        {
            this->grads["dW" + to_string(layer)]->~Matrix();
            this->grads["db" + to_string(layer)]->~Matrix();
        }
    }
}

/**
 * @brief Train neural network with data @param train (create batches)
 * 
 * @param train dataset on which will be network trained
 */
void NeuralNetworkMultiLayer::fit(Dataset *train)
{
    vector<Matrix *> X;
    vector<Matrix *> Y;
    vector<int> batch_index;
    vector<double> costs;
    auto rng = default_random_engine{};
    double acc;
    int offset = 0;
    int length_data = train->getMaxRow();

    // Create batchs from data
    while (length_data != 0)
    {
        if (length_data - this->batchSize < 0)
            this->batchSize = length_data;

        X.push_back(new Matrix(train->getX(), batchSize, offset));
        Y.push_back(new Matrix(train->getY(), batchSize, offset));

        length_data -= this->batchSize;
        offset += this->batchSize;
    }

    // Create array with shuffled batches (array contains shuffled indexes which will be used as index to batch )
    for (int i = 0; i < X.size(); i++)
        batch_index.push_back(i);

    // Initialize weights with random values
    this->initialize();

    for (int epoch = 0; epoch < this->epoch; epoch++)
    {
        // Shuffle batch indexes
        shuffle(batch_index.begin(), batch_index.end(), rng);

        // Get start time of learning
        auto start = high_resolution_clock::now();
        for (int batch = 0; batch < X.size(); batch++)
        {
            Matrix *xBatch = X.at(batch_index.at(batch));
            Matrix *yBatch = Y.at(batch_index.at(batch));

            int batchLength = xBatch->getColumns();

            // Forward batch to get predtion
            this->forwardPropagation(xBatch);
            // Get derivative from predtion
            this->backPropagation(yBatch, batchLength);

            // Update momentum params
            for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
            {
                this->params["V_dW" + to_string(layer)] = momentumSum(this->params["V_dW" + to_string(layer)], this->beta, this->grads["dW" + to_string(layer)], (1.0 - this->beta));
                this->params["V_db" + to_string(layer)] = momentumSum(this->params["V_db" + to_string(layer)], this->beta, this->grads["db" + to_string(layer)], (1.0 - this->beta));
            }

            // Update weights
            for (int layer = 1; layer <= (this->layer.size() - 1); layer++)
            {
                this->params["W" + to_string(layer)] = momentumUpdate(this->params["W" + to_string(layer)], this->params["V_dW" + to_string(layer)], this->learningRate);
                this->params["b" + to_string(layer)] = momentumUpdate(this->params["b" + to_string(layer)], this->params["V_db" + to_string(layer)], this->learningRate);
            }
            // Compute loss function
            costs.push_back(costCrossEntropy(this->cache["A" + to_string(this->layer.size() - 1)], yBatch));

            // Clear cache data from backpropagation and forwardpropagation
            this->clearCache(true);
        }
        auto stop = high_resolution_clock::now();

        // Get mean of every loss function computed from batch
        double cost = std::accumulate(costs.begin(), costs.end(), 0.0) / costs.size();
        cout << "Epoch [" << epoch << "] training cost: " << cost << " training time: " << duration_cast<seconds>(stop - start).count() << " seconds" << endl;
    }
}

/**
 * @brief Make prediction on data, compute accuracy from result and save data into @param fileName file
 * 
 * @param test data to make prediction
 * @param fileName name of file to save predictions
 * @return accuracy of dataset @param test
 */
double NeuralNetworkMultiLayer::transform(Dataset *test, string fileName)
{
    // Make prediction on data
    this->forwardPropagation(test->getX());
    // Transform data to compute accuracy
    double acc = accuracy(this->cache["A" + to_string(this->layer.size() - 1)], test->getY(), fileName);
    // Clear cache from forwardPropagation
    this->clearCache(false);
    return acc;
}