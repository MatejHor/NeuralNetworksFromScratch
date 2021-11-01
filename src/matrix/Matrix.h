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
    double **matrix;
    int rows;
    int columns;

public:
    Matrix(int rows, int columns);
    Matrix(int rows, int columns, double seed);
    Matrix(int rows, int columns, double **other);
    Matrix(const Matrix &other);
    Matrix(Matrix *other, int batchSize);
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

    bool operator==(const Matrix &other) const;
    bool operator!=(const Matrix &other) const;
    double **copyMatrix(double** _matrix, int length);
    void setMatrix(double **_matrix);
    void print();
    void print(string name);
    void printParams();
    void printParams(string name);
    Matrix *T();
};

#endif