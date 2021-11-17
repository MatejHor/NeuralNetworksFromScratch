#include "MatrixOperation.h"
#include <math.h>

/**
 * @brief sum operation for two matrix (ordinaly sum of matrix)
 * 
 * @param m1 first matrix to sum
 * @param m2 second matrix to sum
 * @return new summed Matrix* 
 */
static Matrix *sum(Matrix *m1, Matrix *m2)
{
    if ((m1->getRows() != m2->getRows()) || (m1->getColumns() != m2->getColumns()))
    {
        cout << "[-] CAN NOT SUM MATRIX(NOT HAVE SHAPE TO SUM) m1&=" << (m1) << " m2&=" << (m2) << endl;
        return NULL;
    }

    int rows = m1->getRows();
    int columns = m1->getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = m1->getMatrix()[row][column] + m2->getMatrix()[row][column];
        }
    }

    return summed;
}


/**
 * @brief Sum matrix and vector, it must have the same shape of row
 * 
 * @param matrix
 * @param vector
 * @return new summed Matrix* 
 */
static Matrix *sumVector(Matrix *matrix, Matrix *vector)
{
    if ((matrix->getRows() != vector->getRows()))
    {
        cout << "[-] CAN NOT SUM VECTOR AND MATRIX(NOT HAVE SHAPE TO SUM) m&=" << (matrix) << " vector&=" << (vector) << endl;
        cout << "Shape(m_rows=" << matrix->getRows() << ", vector_columns=" << vector->getRows() << endl;
        return NULL;
    }

    int rows = matrix->getRows();
    int columns = matrix->getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = matrix->getMatrix()[row][column] + vector->getMatrix()[row][0];
        }
    }

    return summed;
}

/**
 * @brief subtrack operation for two matrix (ordinaly subtrack of matrix)
 * 
 * @param m1 first matrix to subtrack
 * @param m2 second matrix to subtrack
 * @return new subtracked Matrix* 
 */
static Matrix *subtrack(Matrix *m1, Matrix *m2)
{
    if ((m1->getRows() != m2->getRows()) && (m1->getColumns() != m2->getColumns()))
    {
        cout << "[-] CAN NOT SUBTRACK MATRIX(NOT HAVE SHAPE TO SUM) m1&=" << (m1) << " m2&=" << (m2) << endl;
        cout << "Shape(m1_rows=" << m1->getRows() << ", m2_rows=" << m2->getRows() << endl;
        cout << "Shape(m1_columns=" << m1->getColumns() << ", m2_columns=" << m2->getColumns() << endl;
        return NULL;
    }

    int rows = m1->getRows();
    int columns = m1->getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = m1->getMatrix()[row][column] - m2->getMatrix()[row][column];
        }
    }

    return summed;
}

/**
 * @brief Sum every item in matrix with value @param x
 * 
 * @param matrix 
 * @param x value to sum
 * @return new Matrix* 
 */
static Matrix *sum(Matrix *matrix, double x)
{
    int rows = matrix->getRows();
    int columns = matrix->getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = matrix->getMatrix()[row][column] + x;
        }
    }

    return summed;
}

/**
 * @brief Subtract every item in matrix with value @param x
 * 
 * @param matrix 
 * @param x value to subtract
 * @return new Matrix* 
 */
static Matrix *subtrack(double x, Matrix *matrix)
{
    int rows = matrix->getRows();
    int columns = matrix->getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = x - matrix->getMatrix()[row][column];
        }
    }

    return summed;
}

/**
 * @brief Sum dimension in Matrix form NxM matrix create Nx1 Matrix
 * 
 * @param matrix 
 * @return new Nx1 Matrix* 
 */
static Matrix *sumDimension(Matrix *matrix)
{
    double **new_matrix = new double *[matrix->getRows()];
    double **_matrix = matrix->getMatrix();

    for (int row = 0; row < matrix->getRows(); row++)
    {
        new_matrix[row] = new double[1];
        double sum = 0;
        for (int column = 0; column < matrix->getColumns(); column++)
            sum += _matrix[row][column];
        *new_matrix[row] = sum;
    }

    return new Matrix(matrix->getRows(), 1, new_matrix);
}

/**
 * @brief Sum Matrix into value
 * 
 * @param matrix 
 * @return sum of item in matrix (double)
 */
static double sumMatrix(Matrix *matrix)
{
    double **_matrix = matrix->getMatrix();
    double sum = 0;

    for (int row = 0; row < matrix->getRows(); row++)
    {
        for (int column = 0; column < matrix->getColumns(); column++)
            sum += _matrix[row][column];
    }

    return sum;
}

/**
 * @brief Create matrix as dot product of m1 and m2 matrices (ordinaly multiply 2 matrices)
 * 
 * @param m1 
 * @param m2 
 * @return new Matrix* 
 */
static Matrix *dot(Matrix *m1, Matrix *m2)
{
    if ((m1->getColumns() != m2->getRows()))
    {
        cout << "[-] CAN NOT DOT MATRIX(NOT HAVE SHAPE TO SUM) m1&=" << (m1) << " m2&=" << (m2) << " Shape(m1_columns=" << m1->getColumns() << ", m2_rows=" << m2->getRows() << ")" << endl;
        cout << "Shape(m1_rows=" << m1->getRows() << ", m2_columns=" << m2->getColumns() << endl;
        return NULL;
    }

    int rows = m1->getRows();
    int mid = m1->getColumns();
    int cols = m2->getColumns();

    Matrix *multi = new Matrix(rows, cols);
    double element = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < cols; k++)
        {
            element = 0;
            for (int j = 0; j < mid; j++)
            {
                element += m1->getMatrix()[i][j] * m2->getMatrix()[j][k];
            }
            multi->getMatrix()[i][k] = element;
        }
    }

    return multi;
}

/**
 * @brief Multiply 2 matrices (not dot product) multiply every item with the item in same place in second matrix
 * 
 * @param m1 
 * @param m2 
 * @return new Matrix* 
 */
static Matrix *multiply(Matrix *m1, Matrix *m2)
{
    if ((m1->getRows() != m2->getRows()) && (m1->getColumns() != m2->getColumns()))
    {
        cout << "Matrices don't have the same shape! Please enter valid matrices.\n";
        return NULL;
    }

    int rows = m1->getRows();
    int columns = m1->getColumns();
    double **new_matrix = new double *[rows];
    double **matrix_1 = m1->getMatrix();
    double **matrix_2 = m2->getMatrix();

    for (int row = 0; row < rows; row++)
    {
        new_matrix[row] = new double[columns];
        for (int column = 0; column < columns; column++)
            new_matrix[row][column] = matrix_1[row][column] * matrix_2[row][column];
    }

    return new Matrix(rows, columns, new_matrix);
}

/**
 * @brief Divide 2 matrices (not dot product) divide every item with the item in same place in second matrix
 * 
 * @param m1 
 * @param m2 
 * @return new Matrix* 
 */
static Matrix *divide(Matrix *m1, Matrix *m2)
{
    if ((m1->getRows() != m2->getRows()) && (m1->getColumns() != m2->getColumns()))
    {
        cout << "Matrices don't have the same shape! Please enter valid matrices.\n";
        return NULL;
    }

    int rows = m1->getRows();
    int columns = m1->getColumns();
    double **new_matrix = new double *[rows];
    double **matrix_1 = m1->getMatrix();
    double **matrix_2 = m2->getMatrix();

    for (int row = 0; row < rows; row++)
    {
        new_matrix[row] = new double[columns];
        for (int column = 0; column < columns; column++)
            new_matrix[row][column] = matrix_1[row][column] / matrix_2[row][column];
    }

    return new Matrix(rows, columns, new_matrix);
}

/**
 * @brief Multiply every item in matrix with value @param x
 * 
 * @param matrix 
 * @param x 
 * @return new mutliply Matrix* 
 */
static Matrix *multiply(Matrix *matrix, double x)
{
    int rows = matrix->getRows();
    int columns = matrix->getColumns();
    double **new_matrix = new double *[rows];
    double **matrix_1 = matrix->getMatrix();

    for (int row = 0; row < rows; row++)
    {
        new_matrix[row] = new double[columns];
        for (int column = 0; column < columns; column++)
            new_matrix[row][column] = matrix_1[row][column] * x;
    }

    return new Matrix(rows, columns, new_matrix);
}

/**
 * @brief applyed log function for every item in Matrix
 * 
 * @param matrix 
 * @return new logged Matrix* 
 */
static Matrix *log(Matrix *matrix)
{
    double **new_matrix = new double *[matrix->getRows()];
    double **_matrix = matrix->getMatrix();

    for (int row = 0; row < matrix->getRows(); row++)
    {
        new_matrix[row] = new double[matrix->getColumns()];
        for (int column = 0; column < matrix->getColumns(); column++)
            new_matrix[row][column] = log(_matrix[row][column]);
    }

    return new Matrix(matrix->getRows(), matrix->getColumns(), new_matrix);
}

// NEURAL NETWORK OPERATIONS
/**
 * @brief Sum matrices into value and before it multiply 2 matrices
 * 
 * @param m1 
 * @param m2 
 * @return sum of multipled matrices (double)
 */
static double crossEntropySum(Matrix *m1, Matrix *m2)
{
    double sum = 0;

    for (int row = 0; row < m1->getRows(); row++)
    {
        for (int column = 0; column < m1->getColumns(); column++)
            sum += (m1->getMatrix()[row][column] * (log(m2->getMatrix()[row][column])));
    }

    return sum;
}

/**
 * @brief Dot product of 2 matrices and after that add item in same row from vector to it
 * 
 * @param m1 
 * @param m2 
 * @param vector 
 * @return new Matrix* 
 */
static Matrix *forwardDot(Matrix *m1, Matrix *m2, Matrix *vector)
{

    int rows = m1->getRows();
    int mid = m1->getColumns();
    int cols = m2->getColumns();

    Matrix *multi = new Matrix(rows, cols);
    double element = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < cols; k++)
        {
            element = 0;
            for (int j = 0; j < mid; j++)
            {
                element += m1->getMatrix()[i][j] * m2->getMatrix()[j][k];
            }
            multi->getMatrix()[i][k] = element + vector->getMatrix()[i][0];
        }
    }

    return multi;
}

/**
 * @brief Dot product of 2 matrices and after that multiply with value @param multiplicator
 * 
 * @param m1 
 * @param m2 
 * @param vector 
 * @return new dot product Matrix* with multiply it with multiplicator
 */
static Matrix *backwardDotDW(Matrix *m1, Matrix *m2, double multiplicator)
{
    int rows = m1->getRows();
    int mid = m1->getColumns();
    int cols = m2->getRows();

    Matrix *multi = new Matrix(rows, cols);
    double element = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < cols; k++)
        {
            element = 0;
            for (int j = 0; j < mid; j++)
            {
                element += m1->getMatrix()[i][j] * m2->getMatrix()[k][j];
            }
            multi->getMatrix()[i][k] = element * multiplicator;
        }
    }

    return multi;
}


/**
 * @brief Dot product of 2 matrices and after that multiply with matrix derivateSigmoid of @param multiply
 * 
 * @param m1 
 * @param m2 
 * @param vector 
 * @return new dot product Matrix* with multiply it with derivativeSigmoid matrix
 */
static Matrix *backwardDotDZSigmoid(Matrix *m1, Matrix *m2, Matrix *multiply)
{
    int rows = m1->getColumns();
    int mid = m1->getRows();
    int cols = m2->getColumns();

    Matrix *multi = new Matrix(rows, cols);
    double element = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < cols; k++)
        {
            element = 0;
            for (int j = 0; j < mid; j++)
            {
                element += m1->getMatrix()[j][i] * m2->getMatrix()[j][k];
            }
            double s = 1 / (1.0 + exp(-(multiply->getMatrix()[i][k])));
            double sigmoidDerivative = s * (1 - s);
            multi->getMatrix()[i][k] = element * sigmoidDerivative;
        }
    }

    return multi;
}

/**
 * @brief Sum matrix dimension into value and multiply it with @param multiplicator
 * 
 * @param matrix 
 * @param multiplicator 
 * @return new Matrix* of shape Nx1 with multiply it with multiplicator
 */
static Matrix *backwardSumDimension(Matrix *matrix, double multiplicator)
{
    double **new_matrix = new double *[matrix->getRows()];
    double **_matrix = matrix->getMatrix();

    for (int row = 0; row < matrix->getRows(); row++)
    {
        new_matrix[row] = new double[1];
        double sum = 0;
        for (int column = 0; column < matrix->getColumns(); column++)
            sum += _matrix[row][column];
        *new_matrix[row] = sum * multiplicator;
    }

    return new Matrix(matrix->getRows(), 1, new_matrix);
}

/**
 * @brief Update momentum with momentum * beta + derivate of weights * (1 - beta) 
 * 
 * @param m1 momentum params (weight or bias)
 * @param multiplicator1 beta
 * @param m2 grads param (weight or bias)
 * @param multiplicator2 1 - beta
 * @return updated momentum Matrix* ! NOT CREATE NEW MATRICES !
 */
static Matrix *momentumSum(Matrix *m1, double multiplicator1, Matrix *m2, double multiplicator2)
{
    int rows = m1->getRows();
    int columns = m1->getColumns();

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            m1->getMatrix()[row][column] = (m1->getMatrix()[row][column] * multiplicator1) + (m2->getMatrix()[row][column] * multiplicator1);
        }
    }

    return m1;
}

/**
 * @brief Update params (only one at time) with previous params - momentum params * learning rate
 * 
 * @param m1 old weight or bias
 * @param multiplicator learning rate
 * @param m2 momentum value (momentum of weight or bias)
 * @return updated params Matrix* ! NOT CREATE NEW MATRICES !
 */
static Matrix *momentumUpdate(Matrix *m1, Matrix *m2, double multiplicator)
{
    int rows = m1->getRows();
    int columns = m1->getColumns();

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            m1->getMatrix()[row][column] = m1->getMatrix()[row][column] - (m2->getMatrix()[row][column] * multiplicator);
        }
    }

    return m1;
}

// ACTIVACNE FUNKCIE
static Matrix *sigmoid(Matrix *x)
{
    double **matrix = new double *[x->getRows()];
    double **X = x->getMatrix();

    for (int row = 0; row < x->getRows(); row++)
    {
        matrix[row] = new double[x->getColumns()];
        for (int column = 0; column < x->getColumns(); column++)
        {
            matrix[row][column] = 1.0 / (1.0 + exp(-X[row][column]));
        }
    }

    return new Matrix(x->getRows(), x->getColumns(), matrix);
}

static Matrix *sigmoidDerivative(Matrix *x)
{
    double **matrix = new double *[x->getRows()];
    double **X = x->getMatrix();

    for (int row = 0; row < x->getRows(); row++)
    {
        matrix[row] = new double[x->getColumns()];
        for (int column = 0; column < x->getColumns(); column++)
        {
            double s = 1 / (1.0 + exp(-(X[row][column])));
            matrix[row][column] = s * (1.0 - s);
        }
    }

    return new Matrix(x->getRows(), x->getColumns(), matrix);
}

static Matrix *softmax(Matrix *x)
{
    double **matrix = new double *[x->getRows()];
    double **X = x->getMatrix();

    for (int row = 0; row < x->getRows(); row++)
    {
        matrix[row] = new double[x->getColumns()];
    }

    for (int column = 0; column < x->getColumns(); column++)
    {
        double sum = 0.0;
        for (int row = 0; row < x->getRows(); row++)
        {
            sum += exp(X[row][column]);
        }

        for (int row = 0; row < x->getRows(); row++)
        {
            double x_exp = exp(X[row][column]);
            matrix[row][column] = x_exp / sum;
        }
    }

    return new Matrix(x->getRows(), x->getColumns(), matrix);
}

static Matrix *softmaxDerivation(Matrix *x)
{
    double **matrix = new double *[x->getRows()];
    double **X = x->getMatrix();

    for (int row = 0; row < x->getRows(); row++)
    {
        matrix[row] = new double[x->getColumns()];
    }

    for (int column = 0; column < x->getColumns(); column++)
    {
        double sum = 0;
        for (int row = 0; row < x->getRows(); row++)
        {
            sum += exp(X[row][column]);
        }

        for (int row = 0; row < x->getRows(); row++)
        {
            double x_softmax = exp(X[row][column]) / sum;
            matrix[row][column] = x_softmax * (1.0 - x_softmax);
        }
    }

    return new Matrix(x->getRows(), x->getColumns(), matrix);
}

static Matrix *reLu(Matrix *x)
{
    double **matrix = new double *[x->getRows()];
    double **X = x->getMatrix();

    for (int row = 0; row < x->getRows(); row++)
    {
        matrix[row] = new double[x->getColumns()];
        for (int column = 0; column < x->getColumns(); column++)
        {
            if (X[row][column] <= 0)
            {
                matrix[row][column] = 0.0;
            }
            else
            {
                matrix[row][column] = X[row][column];
            }
        }
    }
    return new Matrix(x->getRows(), x->getColumns(), matrix);
}

static Matrix *reLuDerivation(Matrix *x)
{
    double **matrix = new double *[x->getRows()];
    double **X = x->getMatrix();

    for (int row = 0; row < x->getRows(); row++)
    {
        matrix[row] = new double[x->getColumns()];
        for (int column = 0; column < x->getColumns(); column++)
        {
            if (X[row][column] <= 0)
            {
                matrix[row][column] = 0.0;
            }
            else
            {
                matrix[row][column] = 1.0;
            }
        }
    }
    return new Matrix(x->getRows(), x->getColumns(), matrix);
}

static Matrix *squeeze(Matrix *Y, string func)
{
    Matrix *new_Y = new Matrix(Y->getRows(), 1);
    for (int row = 0; row < Y->getRows(); row++)
    {
        double comperator = 0.0;
        double foundedValue = 0.0;
        for (int column = 0; column < Y->getColumns(); column++)
        {
            if (func.compare("max") == 0 && Y->getMatrix()[row][column] > comperator)
            {
                comperator = Y->getMatrix()[row][column];
                foundedValue = column * 1.0;
            }
            if (func.compare("category") == 0 && Y->getMatrix()[row][column] == 1.0)
            {
                foundedValue = column * 1.0;
            }
        }
        new_Y->getMatrix()[row][0] = foundedValue;
    }
    return new_Y;
}

void savePrediction(string fileName, Matrix* data, string mode) {
    ofstream file;
    if (mode.compare("append") == 0)
        file.open(fileName, std::ios_base::app);
    else 
        file.open(fileName);

    for (int row = 0; row < data->getRows(); row++)
         for (int column = 0; column < data->getColumns(); column++)
            file << data->getMatrix()[row][column] << "\n";

    file.close();
}

static double accuracy(Matrix *AL, Matrix *Y, string fileName)
{
    Matrix *transponseAL = AL->T();
    Matrix *transponseY = Y->T();
    Matrix *AL_squeeze = squeeze(transponseAL, "max");
    Matrix *Y_squeeze = squeeze(transponseY, "category");

    savePrediction(fileName, AL_squeeze, "create");

    double **al = AL_squeeze->getMatrix();
    double **y = Y_squeeze->getMatrix();


    double TP = 0;

    for (int row = 0; row < transponseAL->getRows(); row++)
    {
        (al[row][0] == y[row][0] && ++TP);
    }

    transponseAL->~Matrix();
    transponseY->~Matrix();
    AL_squeeze->~Matrix();
    Y_squeeze->~Matrix();
    return TP / Y->getColumns();
}