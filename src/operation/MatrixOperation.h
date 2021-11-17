#ifndef MATRIX_OPERATION_H_
#define MATRIX_OPERATION_H_

#include "../matrix/Matrix.h"
#include <math.h>

/**
 * @brief Use as static operation to work with Matrices
 * In network we use only function from section:
 * 1. **FUNCTION FOR NEURAL NETWORK** - 
 *      which are speed up function only for one usage
 *      (they contain more operation as multiply dot log sum...)
 * 2. **ACTIVATION FUNCTION AND THEIR DERIVATIVE** - sigmoid and softmax,
 * 3. **UPDATE PRED MATRIX** - all functions to work with prediction and lables
 * 
 * Also every function are immutable so it create new Matrix 
 * and return address of matrix.
 */
class MatrixOperation
{
public:
    static Matrix *sum(Matrix *m1, Matrix *m2);
    static Matrix *sum(Matrix *matrix, double x);
    static Matrix *sumVector(Matrix *matrix, Matrix *vector);
    static Matrix *subtrack(Matrix *m1, Matrix *m2);
    static Matrix *subtrack(double x, Matrix *matrix);
    static double sumMatrix(Matrix *matrix);
    static Matrix *sumDimension(Matrix *matrix);
    static Matrix *dot(Matrix *m1, Matrix *m2);
    static Matrix *multiply(Matrix *m1, Matrix *m2);
    static Matrix *divide(Matrix *m1, Matrix *m2);
    static Matrix *multiply(Matrix *matrix, double x);
    static Matrix *log(Matrix *matrix);

    // FUNCTION FOR NEURAL NETWORK
    static double crossEntropySum(Matrix *m1, Matrix *m2);
    static Matrix* forwardDot(Matrix* m1, Matrix* m2, Matrix* vector);
    static Matrix* backwardDotDW(Matrix* m1, Matrix* m2, double multiplicator);
    static Matrix *backwardDotDZSigmoid(Matrix* m1, Matrix* m2, Matrix* multiply);
    static Matrix *backwardSumDimension(Matrix *matrix, double multiplicator);
    static Matrix *momentumSum(Matrix *m1, double multiplicator1, Matrix *m2, double multiplicator2);
    static Matrix *momentumUpdate(Matrix *m1, Matrix *m2, double multiplicator);

    // ACTIVATION FUNCTION AND THEIR DERIVATIVE
    static Matrix *sigmoid(Matrix *x);
    static Matrix *sigmoidDerivative(Matrix *x);
    static Matrix *softmax(Matrix *x);
    static Matrix *softmaxDerivation(Matrix *x);
    static Matrix *reLu(Matrix *x);
    static Matrix *reLuDerivation(Matrix *x);

    // UPDATE PREDICTION MATRIX
    void savePrediction(string fileName, Matrix* data, string mode);
    static Matrix *squeeze(Matrix *Y, string func);
    static double accuracy(Matrix *AL, Matrix *Y);
};

#endif