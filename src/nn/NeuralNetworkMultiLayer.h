#ifndef NeuralNetworkMultiLayer_H_
#define NeuralNetworkMultiLayer_H_

#include "../matrix/Matrix.h"
#include "../operation/MatrixOperation.h"
#include "../dataset/Dataset.h"

// MEAN for LOSS
#include <numeric>
// FILE
#include <iostream>
// SHUFFLE Batches
#include <algorithm>
// TIME
#include <chrono>

#include <vector>
#include <map>
#include <random>
using namespace chrono;

class NeuralNetworkMultiLayer
{
private:
    // Number of epoch which will be network trained
    int epoch = 20;
    // Size of batch for data
    int batchSize = 256;
    // Learning rate for update weight and bias
    double learningRate = 4.0;
    // Beta parameter for momentum
    double beta = 0.9;

    // Hierarchy of network, number of neurons in layer
    vector<int> layer;
    // Hash map for saving Matrices from ForwardPropagation
    map<string, Matrix *> cache;
    // Hash map for saving Matrices from BackPropagation
    map<string, Matrix *> grads;
    // Hash map for saving Weights, Biases and Momentum
    map<string, Matrix *> params;

public:
    NeuralNetworkMultiLayer(vector<int> layer, int epochs, int batchSize, double learningRate, double beta);

    void initialize();
    void forwardPropagation(Matrix *xBatch);
    double costCrossEntropy(Matrix *AL, Matrix *yBatch);
    void backPropagation(Matrix *yBatch, double m_batch);

    void fit(Dataset *train);
    double transform(Dataset *test, string fileName);
    void clearCache(bool clearGrads);

    // Dealocate params
    ~NeuralNetworkMultiLayer()
    {
        for (int layer = 1; layer < (this->layer.size()); layer++)
        {
            params["W" + to_string(layer)]->~Matrix();
            params["b" + to_string(layer)]->~Matrix();

            params["V_dW" + to_string(layer)]->~Matrix();
            params["V_db" + to_string(layer)]->~Matrix();
        }
    }
};

#endif