#ifndef MATRIX_OPERATION_H_
#define MATRIX_OPERATION_H_

#include "../matrix/Matrix.h"
#include <math.h> 


class MatrixOperation
{
public:
    static Matrix* sum(Matrix m1, Matrix m2);
    static Matrix* sum(Matrix m, double x);
    static double sumMatrix(Matrix m);
    static Matrix* sumDimension(Matrix m);
    static Matrix* dot(Matrix m1, Matrix m2);
    static Matrix* multiply(Matrix m1, Matrix m2);
    static Matrix *multiply(Matrix m, double x);
    static Matrix* log(Matrix m);

    static Matrix* sigmoid(Matrix x);
    static Matrix* sigmoidDerivative(Matrix x);
    static Matrix* softmax(Matrix x);
    static Matrix* softmaxDerivation(Matrix x);
    static Matrix* reLu(Matrix x);
    static Matrix* reLuDerivation(Matrix x);
};

#endif