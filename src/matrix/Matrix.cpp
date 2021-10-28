#include "Matrix.h"

Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns)
{
    double** new_matrix = new double*[rows];
    for (int row = 0; row < rows; row++)
    {
        new_matrix[row] = new double[columns];
    }

    this->matrix = new_matrix;
}

Matrix::Matrix()
{
}

Matrix::Matrix(int rows, int columns, double seed): rows(rows), columns(columns)
{
    // unsigned seed = 1;
    // std::mt19937 generator1(1);
    // static_cast<double>(generator1()) / numeric_limits<uint32_t>::max();

    // srand(seed * 50684764);
    // std::default_random_engine generator();
    // std::chrono::system_clock::now().time_since_epoch().count();
    // std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    // std::normal_distribution<double> distribution(0.0, 1.0);

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};

    double** new_matrix = new double*[rows];
    for (int row = 0; row < rows; row++)
    {
        new_matrix[row] = new double[columns];
        for (int column = 0; column < columns; column++) {
            
            new_matrix[row][column] = d(gen) * seed;
        }
    }

    this->matrix = new_matrix;
}

Matrix::Matrix(int rows, int columns, double **other) : rows(rows), columns(columns)
{
    matrix = other;
    cout << "Creating Matrix(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
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
    cout << "Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl;
    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            cout << matrix[row][column] << " ";
        }

        cout << endl;
    }
}

void Matrix::printParams()
{
    cout << "Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl;
}

void Matrix::setMatrix(double** _matrix) {
    matrix = this->copyMatrix(_matrix);
}

Matrix* Matrix::T()
{
    cout << "Transpose matrix(old&=" << (this);
    Matrix* transposition = new Matrix(columns, rows);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            transposition->matrix[column][row] = matrix[row][column];
        }
    }
    cout << ",new&=" << transposition << ")" << endl;
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

