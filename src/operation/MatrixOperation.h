#ifndef MATRIX_OPERATION_H_
#define MATRIX_OPERATION_H_

#include "../matrix/Matrix.h"
#include <math.h>

class MatrixOperation
{
public:
    static Matrix *sum(Matrix *m1, Matrix *m2);
    static Matrix *sum(Matrix *m, double x);
    static Matrix *sumVector(Matrix *m, Matrix *vector);
    static Matrix *subtrack(Matrix *m1, Matrix *m2);
    static Matrix *subtrack(double x, Matrix *m);
    static double sumMatrix(Matrix *m);
    static Matrix *sumDimension(Matrix *m);
    static Matrix *dot(Matrix *m1, Matrix *m2);
    static Matrix *multiply(Matrix *m1, Matrix *m2);
    static Matrix *divide(Matrix *m1, Matrix *m2);
    static Matrix *multiply(Matrix *m, double x);
    static Matrix *log(Matrix *m);

    // FUNCTION FOR NEURAL NETWORK
    static double crossEntropySum(Matrix *m1, Matrix *m2);
    static Matrix* forwardDot(Matrix* m1, Matrix* m2, Matrix* vector);
    static Matrix* backwardDotDW(Matrix* m1, Matrix* m2, double multiplicator);
    static Matrix *backwardDotDZSigmoid(Matrix* m1, Matrix* m2, Matrix* multiply);
    static Matrix *backwardSumDimension(Matrix *m, double multiplicator);
    static Matrix *momentumSum(Matrix *m1, double multiplicator1, Matrix *m2, double multiplicator2);
    static Matrix *momentumUpdate(Matrix *m1, Matrix *m2, double multiplicator);

    // ACTIVATION FUNCTION AND DERIVATIVE
    static Matrix *sigmoid(Matrix *x);
    static Matrix *sigmoidDerivative(Matrix *x);
    static Matrix *softmax(Matrix *x);
    static Matrix *softmaxDerivation(Matrix *x);
    static Matrix *reLu(Matrix *x);
    static Matrix *reLuDerivation(Matrix *x);

    // UPDATE PRED MATRIX
    static Matrix *squeeze(Matrix *Y, string func);
    static double accuracy(Matrix *AL, Matrix *Y);
};

#endif