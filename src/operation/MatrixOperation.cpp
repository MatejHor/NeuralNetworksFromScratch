#include "MatrixOperation.h"
#include "Matrix.h"

static Matrix *sum(Matrix m1, Matrix m2)
{
    if ((m1.getRows() == m2.getRows()) && (m1.getColumns() == m2.getColumns()))
    {
        cout << "Matrices don't have the same shape! Please enter valid matrices.\n";
        return;
    }

    int rows = m1.getRows();
    int columns = m2.getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = m1.getMatrix()[row][column] + m2.getMatrix()[row][column];
        }
    }

    return summed;
}

static Matrix sumDimension(Matrix m1, Matrix m2)
{
}

static Matrix dot(Matrix m1, Matrix m2)
{
}

static Matrix *multiply(Matrix m1, Matrix m2)
{
    if ((m1.getColumns() != m2.getRows()))
    {
        cout << "Matrices can't be multiplied!\nPlease enter valid matricies for multiplication.\n";
        return;
    }

    int rows = m1.getRows();
    int mid = m1.getColumns();
    int cols = m2.getColumns();

    Matrix *multi = new Matrix(rows, cols);
    double element = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < cols; k++)
        {
            element = 0;
            for (int j = 0; j < mid; j++)
            {
                element += m1.getMatrix()[i][j] * m2.getMatrix()[j][k];
            }
            multi->getMatrix()[i][k] = element;
        }
    }

    return multi;
}

static Matrix log(Matrix m)
{
}
