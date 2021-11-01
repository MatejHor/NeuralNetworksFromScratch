#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "../matrix/Matrix.h"
#include "../operation/MatrixOperation.h"
#include "../dataset/Dataset.h"

// SIZE of double[]
#include <bits/stdc++.h>

#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>

// TIME
#include <chrono>
using namespace chrono;

class NeuralNetwork
{
private:
    double* layer_dims;
    int len_layer;
    int epochs;
    int batchSize;
    double learning_rate;
    double accuracy;
    bool VERBOSE;

    Matrix *AL = NULL;
    vector<Matrix *> A_cache;
    vector<Matrix *> dA_cache;
    vector<Matrix *> dW_cache;
    vector<Matrix *> db_cache;

    vector<Matrix *> xBatch;
    vector<Matrix *> yBatch;
    int batchCount;

    vector<Matrix *> weights;
    vector<Matrix *> bias;
public:
    NeuralNetwork(double* layer_dims, int epochs, int batchSize, double learning_rate, bool verbose);

    ~NeuralNetwork() {
        freeCache();
        freeMatrixVector(xBatch, yBatch, batchCount);
    }
    void initializeRandomParameters();

    vector<Matrix *> initializeBias(double layer_dims[]);
    vector<Matrix *> initializeRandomWeights(double layer_dims[]);
    vector<Matrix *> initializeHeWeights(double layer_dims[]);

    int initializeMiniBatch(vector<Matrix *> &X, vector<Matrix *> &Y, Dataset *data, int batchSize);

    Matrix *forwardPropagation(Matrix *X, vector<Matrix *> weights, vector<Matrix *> bias);
    void backwardPropagation(Matrix *AL, Matrix *Y, vector<Matrix *> weights);
    
    double computeCostCrossEntropy(Matrix *AL, Matrix *Y);

    void updateParametersGradientDescend(vector<Matrix *> &bias, vector<Matrix *> &weights);

    void fit(Dataset *train, Dataset *validation);
    double predict(Dataset *data, vector<Matrix *> weights, vector<Matrix *> bias, string measure, bool verbose);

    void saveParameters(string path);

    void freeMatrixVector(vector<Matrix *> &vector1, vector<Matrix *> &vector2, int count);
    void freeCache();

};

#endif