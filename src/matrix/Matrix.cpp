#include "Matrix.h"

Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns)
{
    matrix = new double*[rows];
    for (int row = 0; row < rows; row++)
    {
        matrix[row] = new double[columns];
    }
}

Matrix::Matrix(int rows, int columns, double **other) : rows(rows), columns(columns)
{
    matrix = other;
}

Matrix::Matrix(const Matrix &other)
{
    rows = other.rows;
    columns = other.columns;
    matrix = other.matrix;
}

bool Matrix::operator==(const Matrix &other) const
{
    if ((rows == other.rows) && (columns == other.columns))
    {
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                if (matrix[row][column] != other.matrix[row][column])
                    return false;
            }
        }
    }
    else
        return false;
    return true;
}

bool Matrix::operator!=(const Matrix &other) const
{
    if ((rows == other.rows) && (columns == other.columns))
    {
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                if (matrix[row][column] != other.matrix[row][column])
                    return true;
            }
        }
    }
    else
        return true;
    return false;
}

// Matrix Matrix::operator*() const {}

void Matrix::print()
{
    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            std::cout << matrix[row][column];
        }

        std::cout << "\n";
    }
}

Matrix Matrix::T()
{
    Matrix transposition = Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            transposition.matrix[column][row] = matrix[row][column];
        }
    }

    this->~Matrix();
    return transposition;
}

const Matrix &Matrix::operator=(const Matrix &other)
{
    this->rows = other.getRows();
    this->columns = other.getColumns();
    double** _matrix = other.getMatrix();
    double** new_matrix = new double*[rows];

    for (int row = 0; row < rows; row++) {
        new_matrix[row] = new double[columns];
        for (int column = 0; column < columns; column++)
            new_matrix[row][column] = _matrix[row][column];
    }
    this->matrix = new_matrix;

    return *this;
}