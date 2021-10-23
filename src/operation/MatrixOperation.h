#ifndef MATRIX_OPERATION_H_
#define MATRIX_OPERATION_H_

#include "../matrix/Matrix.h"
#include <math.h> 


class MatrixOperation
{
public:
    static Matrix* sum(Matrix m1, Matrix m2);
    static Matrix* sumDimension(Matrix m);
    static Matrix* dot(Matrix m1, Matrix m2);
    static Matrix* multiply(Matrix m1, Matrix m2);
    static Matrix* log(Matrix m);
};

#endif