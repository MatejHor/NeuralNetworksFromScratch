#ifndef MATRIX_H_
#define MATRIX_H_

#include <chrono>
#include <random>
#include <iomanip>

#include <iostream>
#include <stdlib.h> // rand
using namespace std;

class Matrix
{
private:
    // dynamic 2D array holding Matrix values
    double **matrix;
    // number of rows of the Matrix
    int rows;
    // number of columns of the Matrix
    int columns;

public:
    Matrix(int rows, int columns);
    Matrix(int rows, int columns, double seed);
    Matrix(int rows, int columns, double **other);
    Matrix(const Matrix &other);
    Matrix(Matrix *other, int batchSize);
    Matrix(Matrix *other, int batchSize, int offSet);
    Matrix();

    ~Matrix()
    {
        // cout << "Destructing Matrix(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
        
        for (int i = 0; i < rows; i++)
            delete[] matrix[i];
        delete[] matrix;

    }

    int getRows() const
    {
        return rows;
    }

    int getColumns() const
    {
        return columns;
    }

    double **getMatrix() const
    {
        return matrix;
    }
    double **copyMatrix(double** _matrix, int length);
    double **copyMatrixRandom(double** _matrix, int length, int offSet);

    bool operator==(const Matrix &other) const;
    bool operator!=(const Matrix &other) const;
    void setMatrix(double **_matrix);
    Matrix *T();
    
    void print();
    void print(string name);
    void printParams();
    void printParams(string name);

    
};

#endif