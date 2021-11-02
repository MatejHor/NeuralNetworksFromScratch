#include "Matrix.h"

Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns)
{
    double** new_matrix = new double*[rows];
    for (int row = 0; row < rows; row++)
    {
        new_matrix[row] = new double[columns];
    }

    this->matrix = new_matrix;
    // cout << "Creating Matrix(rows, columns)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
}

Matrix::Matrix()
{
    // cout << "Creating Matrix()" << endl;
}

Matrix::Matrix(int rows, int columns, double seed): rows(rows), columns(columns)
{
    random_device rd{};
    mt19937 gen{rd()};
    normal_distribution<> d{0, 1};

    double** new_matrix = new double*[rows];
    for (int row = 0; row < rows; row++)
    {
        new_matrix[row] = new double[columns];
        for (int column = 0; column < columns; column++) {
            
            new_matrix[row][column] = d(gen) * seed;
        }
    }

    this->matrix = new_matrix;
    // cout << "Creating Matrix(rows, columns, seed)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
}

Matrix::Matrix(int rows, int columns, double **other) : rows(rows), columns(columns)
{
    matrix = other;
    // cout << "Creating Matrix(rows, columns, other)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
}

Matrix::Matrix(const Matrix &other)
{
    rows = other.getRows();
    columns = other.getColumns();
    matrix = this->copyMatrix(other.getMatrix(), columns);
    // cout << "Creating Matrix(rows, columns, other)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
}

Matrix::Matrix(Matrix *other, int batchSize)
{
    rows = other->getRows();
    columns = batchSize;
    matrix = this->copyMatrix(other->getMatrix(), columns);
    // cout << "Creating Matrix(other, batchSize)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
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

// const void Matrix::operator=(const Matrix &other) {
//     for(int i = 0; i < this->rows; i++)
//             delete [] this->matrix[i];
//     delete [] this->matrix;

//     this->rows = other.rows;
//     this->columns = other.columns;
//     this->setMatrix(other.getMatrix());
// }

void Matrix::print()
{
    cout << "Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl << "[\n";
    for (int row = 0; row < rows; row++)
    {
        cout << " [ ";
        for (int column = 0; column < columns; column++)
        {
            cout << std::fixed << std::setprecision(8) << matrix[row][column] << " ";
        }

        cout << "]," << endl;
    }
    cout << "]" << endl;
}

void Matrix::print(string name)
{
    cout << name << " Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl << "[\n";
    for (int row = 0; row < rows; row++)
    {
        cout << " [ ";
        for (int column = 0; column < columns; column++)
        {
            cout << std::fixed << std::setprecision(8) << matrix[row][column] << " ";
        }

        cout << "]," << endl;
    }
    cout << "]" << endl;
}

void Matrix::printParams()
{
    cout << "Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl;
}

void Matrix::printParams(string name)
{
    cout << name << " Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl;
}

void Matrix::setMatrix(double** _matrix) {
    matrix = this->copyMatrix(_matrix, this->getColumns());
}

Matrix* Matrix::T()
{
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

double** Matrix::copyMatrix(double** _matrix, int length)
{
    double** new_matrix = new double*[rows];

    for (int row = 0; row < rows; row++) {
        new_matrix[row] = new double[length];
        for (int column = 0; column < length; column++)
            new_matrix[row][column] = _matrix[row][column];
    }

    return new_matrix;
} 

