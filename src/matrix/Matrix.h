#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
using namespace std;

class Matrix
{
private:
    double **matrix;
    int rows;
    int columns;

public:
    Matrix(int rows, int columns);
    Matrix(int rows, int columns, double **other);
    Matrix(const Matrix &other);

    ~Matrix()
    {
        for (int i = 0; i < rows; i++)
            delete[] matrix[i];
        delete[] matrix;

        cout << "Destructing Matrix(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
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
    const void operator=(const Matrix &other);

    double **copyMatrix(double **_matrix);
    void setMatrix(double **_matrix);
    void print();
    Matrix *T();
};

#endif