#ifndef MATRIX_OPERATION_H_
#define MATRIX_OPERATION_H_

#include "../matrix/Matrix.h"
#include <math.h>

class MatrixOperation
{
public:
    static Matrix *sum(Matrix m1, Matrix m2);
    static Matrix *subtrack(Matrix m1, Matrix m2);
    static Matrix *sum(Matrix m, double x);
    static Matrix *subtrack(double x, Matrix m);
    static double sumMatrix(Matrix m);
    static Matrix *sumDimension(Matrix m);
    static Matrix *dot(Matrix m1, Matrix m2);
    static Matrix *multiply(Matrix m1, Matrix m2);
    static Matrix *divide(Matrix m1, Matrix m2);
    static Matrix *multiply(Matrix m, double x);
    static Matrix *log(Matrix m);

    // ACTIVATION FUNCTION AND DERIVATIVE
    static Matrix *sigmoid(Matrix x);
    static Matrix *sigmoidDerivative(Matrix x);
    static Matrix *softmax(Matrix x);
    static Matrix *softmaxDerivation(Matrix x);
    static Matrix *reLu(Matrix x);
    static Matrix *reLuDerivation(Matrix x);
};

#endif