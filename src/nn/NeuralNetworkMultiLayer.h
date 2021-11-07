#ifndef NeuralNetworkMultiLayer_H_
#define NeuralNetworkMultiLayer_H_

#include "../matrix/Matrix.h"
#include "../operation/MatrixOperation.h"
#include "../dataset/Dataset.h"

// SIZE of double[]
#include <bits/stdc++.h>

#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <map>
#include <algorithm>
#include <random>

// TIME
#include <chrono>
using namespace chrono;

class NeuralNetworkMultiLayer
{
private:
    int epoch = 20;
    int batchSize = 256;
    double learningRate = 4.0;
    double beta = 0.9;
    vector<int> layer;
    map<string, Matrix*> cache;
    map<string, Matrix*> grads;
    map<string, Matrix*> params;
public:
    NeuralNetworkMultiLayer(vector<int> layer, int epochs, int batchSize, double learningRate, double beta);

    void initialize();
    void forwardPropagation(Matrix* xBatch);
    double costCrossEntropy(Matrix* AL, Matrix* yBatch);
    void backPropagation(Matrix *xBatch, Matrix *yBatch, double m_batch);

    void fit(Dataset* train);
    double transform(Dataset* test);
    void clearCache();

    ~NeuralNetworkMultiLayer() {
        params["W1"]->~Matrix();
        params["b1"]->~Matrix();
        params["W2"]->~Matrix();
        params["b2"]->~Matrix();

        params["V_dW1"]->~Matrix();
        params["V_db1"]->~Matrix();
        params["V_dW2"]->~Matrix();
        params["V_db2"]->~Matrix();
    }
};

#endif