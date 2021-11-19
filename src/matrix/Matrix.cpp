#include "Matrix.h"

/**
 * @brief Construct a new Matrix:: Matrix object, alocation of dynamic 2D array
 * 
 * @param rows number of rows for the matrix
 * @param columns number of columns for the matrix
 */
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

/**
 * @brief Construct a new Matrix:: Matrix object
 * 
 */
Matrix::Matrix()
{
    // cout << "Creating Matrix()" << endl;
}

/**
 * @brief Construct a new Matrix:: Matrix object, matrix filled with random data or zeros, when seed = 0.
 * 
 * @param rows number of rows for the matrix
 * @param columns number of columns for the matrix
 * @param seed modifier of the generated numbers, provides zero values in matrix when value 0.
 */
Matrix::Matrix(int rows, int columns, double seed): rows(rows), columns(columns)
{
    random_device rd{};
    // mt19937 gen{rd()};
    // mt19937 gen{4089354279};
    mt19937 gen{138};
    // cout << "gen " << gen;
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

/**
 * @brief Construct a new Matrix:: Matrix object, gets 2D array of values
 * 
 * @param rows number of rows for the matrix
 * @param columns number of columns for the matrix
 * @param other 2D array of values
 */
Matrix::Matrix(int rows, int columns, double **other) : rows(rows), columns(columns)
{
    matrix = other;
    // cout << "Creating Matrix(rows, columns, other)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
}

/**
 * @brief Construct a new Matrix:: Matrix object, copies given Matrix
 * 
 * @param other Matrix object we want to copy
 */
Matrix::Matrix(const Matrix &other)
{
    rows = other.getRows();
    columns = other.getColumns();
    matrix = this->copyMatrix(other.getMatrix(), columns);
    // cout << "Creating Matrix(rows, columns, other)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
}

/**
 * @brief Construct a new Matrix:: Matrix object, copies given Matrix with batch size, constructs "submatrix"
 * 
 * @param other Matrix object we want to copy
 * @param batchSize limiter for number of columns we want to copy
 */
Matrix::Matrix(Matrix *other, int batchSize)
{
    rows = other->getRows();
    columns = batchSize;
    matrix = this->copyMatrix(other->getMatrix(), columns);
    // cout << "Creating Matrix(other, batchSize)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
}

/**
 * @brief Construct a new Matrix:: Matrix object, copies given Matrix with batch size and offset, constructs "submatrix"
 * 
 * @param other Matrix object we want to copy
 * @param batchSize limiter for number of columns we want to copy
 * @param offSet index of columns from where we want to copy content of the given Matrix
 */
Matrix::Matrix(Matrix *other, int batchSize, int offSet)
{
    rows = other->getRows();
    columns = batchSize;
    matrix = this->copyMatrixRandom(other->getMatrix(), columns, offSet);
    // cout << "Creating Matrix(other, batchSize)(row=" << rows << ", column=" << columns << ", &=" << (this) << ")" << endl;
}

/**
 * @brief operator for equality of two Matrix objects
 * 
 * @param other second matrix with which the first one will be compared
 * @return true if the two Matrices are equal
 * @return false of the two matrixes are not equal
 */
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

/**
 * @brief operator for inequality of two Matrix objects
 * 
 * @param other second matrix with which the first one will be compared
 * @return true if the two Matrices are not equal
 * @return false of the two matrixes are equal
 */
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

/**
 * @brief prints content of matrix object in structured form
 * 
 */
void Matrix::print()
{
    cout << "Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl << "[\n";
    for (int row = 0; row < rows; row++)
    {
        cout << " [ ";
        for (int column = 0; column < columns; column++)
        {
            cout << std::fixed << std::setprecision(8) << matrix[row][column] << ", ";
        }

        if (row + 1 == rows)
            cout << "]" << endl;
        else
            cout << "]," << endl;
    }
    cout << "]" << endl;
}

/**
 * @brief prints content of matrix object in structured form plus the additional information about the Matrix
 * 
 * @param name additional info about Matrix to print
 */
void Matrix::print(string name)
{
    cout << name << " Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl << "[\n";
    for (int row = 0; row < rows; row++)
    {
        cout << " [ ";
        for (int column = 0; column < columns; column++)
        {
            cout << std::fixed << std::setprecision(8) << matrix[row][column];

            if (column + 1 == columns)
                cout << " ";
            else
                cout << ", ";
        }

        if (row + 1 == rows)
            cout << "]" << endl;
        else
            cout << "]," << endl;
    }
    cout << "]" << endl;
}

/**
 * @brief print parameters of Matrix, consisting of number of rows, columns and its address
 * 
 */
void Matrix::printParams()
{
    cout << "Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl;
}

/**
 * @brief print parameters of Matrix, consisting of number of rows, columns and its address plus additional information about the Matrix
 * 
 * @param name additional info about Matrix to print
 */
void Matrix::printParams(string name)
{
    cout << name << " Matrix(rows=" << rows << ", columns=" << columns << ", &=" << (this) << ")" << endl;
}

/**
 * @brief sets Matrix matrix 2D array
 * 
 * @param _matrix 2D array with values
 */
void Matrix::setMatrix(double** _matrix) {
    matrix = this->copyMatrix(_matrix, this->getColumns());
}

/**
 * @brief standard transposition of the Matrix
 * 
 * @return Matrix* transposed Matrix
 */
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

/**
 * @brief creates new dynamic 2D array and copies the values from given one
 * 
 * @param _matrix 2D array which values we want to copy
 * @param length number of columns we want to copy
 * @return double** newly created 2D array with copied values
 */
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

/**
 * @brief creates new dynamic 2D array and copies subset of values from given one
 * 
 * @param _matrix 2D array from which values we want to copy values
 * @param length number of columns we want to copy
 * @param offSet index of columns from where we want to copy content of the given Matrix
 * @return double** newly created 2D array with copied values
 */
double** Matrix::copyMatrixRandom(double** _matrix, int length, int offSet)
{
    double** new_matrix = new double*[rows];

    for (int row = 0; row < rows; row++) {
        new_matrix[row] = new double[length];
        for (int column = 0; column < length; column++)
            new_matrix[row][column] = _matrix[row][column + offSet];
    }

    return new_matrix;
} 
