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
    int epoch = 20;
    int batchSize = 256
    double learningRate = 4.0;
    double beta = 0.9;
    map<string, Matrix*> cache;
    map<string, Matrix*> grads;
    map<string, Matrix*> params;
public:
    NeuralNetwork(int epochs, int batchSize, double learningRate, double beta);

    void initialize();
    void forwardPropagation(Matrix* xBatch);
    double costCrossEntropy(Matrix* AL, Matrix* yBatch);
    void backPropagation(Matrix *xBatch, Matrix *yBatch, double m_batch);

    void fit(Dataset* train);
    double transform(Dataset* test);

    ~NeuralNetwork() {
        this->params["W1"]->~Matrix();
        this->params["b1"]->~Matrix();
        this->params["W2"]->~Matrix();
        this->params["b2"]->~Matrix();

        this->params["V_dW1"]->~Matrix();
        this->params["V_db1"]->~Matrix();
        this->params["V_dW2"]->~Matrix();
        this->params["V_db2"]->~Matrix();
    }
};

#endif