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
    rows = other.getRows();
    columns = other.getColumns();
    matrix = this->copyMatrix(other.getMatrix());
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

void Matrix::print()
{
    cout << "Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl;
    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            std::cout << matrix[row][column] << " ";
        }

        std::cout << endl;
    }
}

void Matrix::setMatrix(double** _matrix) {
    matrix = this->copyMatrix(_matrix);
}

Matrix* Matrix::T()
{
    cout << "Transpose matrix(&=" << (this) << ")" << endl;
    Matrix* transposition = new Matrix(columns, rows);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            transposition->matrix[column][row] = matrix[row][column];
        }
    }
    return transposition;
}

double** Matrix::copyMatrix(double** _matrix)
{
    double** new_matrix = new double*[rows];

    for (int row = 0; row < rows; row++) {
        new_matrix[row] = new double[columns];
        for (int column = 0; column < columns; column++)
            new_matrix[row][column] = _matrix[row][column];
    }

    return new_matrix;
} 

const Matrix* &Matrix::operator=(const Matrix &other) {
    for(int i = 0; i < this->rows; i++)
            delete [] this->matrix[i];
    delete [] this->matrix;

    this->rows = other.rows;
    this->columns = other.columns;
    this->setMatrix(other.getMatrix());
}