#ifndef DATASET_H_
#define DATASET_H_

#include "../matrix/Matrix.h"
#include <fstream>
#include <string>
#include <sstream>

class Dataset {
private:
    Matrix *X;
    Matrix *Y;

public:
    Dataset(string vectors, string labels);

    ~Dataset()
    {
        (*X).~Matrix();
        (*Y).~Matrix();
    }

    int getRows(string file);

    int getColumns(string file);

    Matrix* getX() const
    {
        return X;
    }

    Matrix* getY() const
    {
        return Y;
    }

    double **readData(string file, int rows, int columns);
    
    void print(int limit);
};

#endif