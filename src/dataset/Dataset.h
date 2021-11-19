#ifndef DATASET_H_
#define DATASET_H_

#include "../matrix/Matrix.h"
#include "../operation/MatrixOperation.h"
#include <fstream>
#include <string>
#include <sstream>

class Dataset
{
private:
    // Matrix for values representing vectors
    Matrix *X;
    // Matrix for values representing labels
    Matrix *Y;
    // number of rows
    int maxRow;

public:
    Dataset(string vectors, string labels, int maxRow, bool verbose);

    ~Dataset()
    {
        X->~Matrix();
        Y->~Matrix();
    }

    Matrix *getX() const
    {
        return X;
    }

    Matrix *getY() const
    {
        return Y;
    }

    int getMaxRow() const
    {
        return maxRow;
    }

    int getRows(string file);
    int getColumns(string file);
    double **readData(string file, int rows, int columns, bool X);
    void print(int limit);
    // double f1_mikro(Matrix *AL);
    // double accuracy(Matrix *AL);
};

#endif