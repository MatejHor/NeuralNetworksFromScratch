#include "MatrixOperation.h"
#include <math.h>

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

static Matrix *sumVector(Matrix *m, Matrix *vector)
{
    if ((m->getRows() != vector->getRows()))
    {
        cout << "[-] CAN NOT SUM VECTOR(NOT HAVE SHAPE TO SUM) m&=" << (m) << " vector&=" << (vector) << endl;
        cout << "Shape(m_rows=" << m->getRows() << ", vector_columns=" << vector->getRows() << endl;
        return NULL;
    }

    int rows = m->getRows();
    int columns = m->getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = m->getMatrix()[row][column] + vector->getMatrix()[row][0];
        }
    }

    return summed;
}

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

static Matrix *sum(Matrix *m, double x)
{
    int rows = m->getRows();
    int columns = m->getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = m->getMatrix()[row][column] + x;
        }
    }

    return summed;
}

static Matrix *subtrack(double x, Matrix *m)
{
    int rows = m->getRows();
    int columns = m->getColumns();

    Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            summed->getMatrix()[row][column] = x - m->getMatrix()[row][column];
        }
    }

    return summed;
}

static Matrix *sumDimension(Matrix *m)
{
    double **new_matrix = new double *[m->getRows()];
    double **_matrix = m->getMatrix();

    for (int row = 0; row < m->getRows(); row++)
    {
        new_matrix[row] = new double[1];
        double sum = 0;
        for (int column = 0; column < m->getColumns(); column++)
            sum += _matrix[row][column];
        *new_matrix[row] = sum;
    }

    return new Matrix(m->getRows(), 1, new_matrix);
}

static double sumMatrix(Matrix *m)
{
    double **_matrix = m->getMatrix();
    double sum = 0;

    for (int row = 0; row < m->getRows(); row++)
    {
        for (int column = 0; column < m->getColumns(); column++)
            sum += _matrix[row][column];
    }

    return sum;
}

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

static Matrix *multiply(Matrix *m, double x)
{
    int rows = m->getRows();
    int columns = m->getColumns();
    double **new_matrix = new double *[rows];
    double **matrix_1 = m->getMatrix();

    for (int row = 0; row < rows; row++)
    {
        new_matrix[row] = new double[columns];
        for (int column = 0; column < columns; column++)
            new_matrix[row][column] = matrix_1[row][column] * x;
    }

    return new Matrix(rows, columns, new_matrix);
}

static Matrix *log(Matrix *m)
{
    double **new_matrix = new double *[m->getRows()];
    double **_matrix = m->getMatrix();

    for (int row = 0; row < m->getRows(); row++)
    {
        new_matrix[row] = new double[m->getColumns()];
        for (int column = 0; column < m->getColumns(); column++)
            new_matrix[row][column] = log(_matrix[row][column]);
    }

    return new Matrix(m->getRows(), m->getColumns(), new_matrix);
}

// NEURAL NETWORK OPERATIONS

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

static Matrix *backwardSumDimension(Matrix *m, double multiplicator)
{
    double **new_matrix = new double *[m->getRows()];
    double **_matrix = m->getMatrix();

    for (int row = 0; row < m->getRows(); row++)
    {
        new_matrix[row] = new double[1];
        double sum = 0;
        for (int column = 0; column < m->getColumns(); column++)
            sum += _matrix[row][column];
        *new_matrix[row] = sum * multiplicator;
    }

    return new Matrix(m->getRows(), 1, new_matrix);
}

static Matrix *momentumSum(Matrix *m1, double multiplicator1, Matrix *m2, double multiplicator2)
{
    int rows = m1->getRows();
    int columns = m1->getColumns();

    // Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            // summed->getMatrix()[row][column] = (m1->getMatrix()[row][column] * multiplicator1) + (m2->getMatrix()[row][column] * multiplicator1);
            m1->getMatrix()[row][column] = (m1->getMatrix()[row][column] * multiplicator1) + (m2->getMatrix()[row][column] * multiplicator1);
        }
    }

    return m1;
}

static Matrix *momentumUpdate(Matrix *m1, Matrix *m2, double multiplicator)
{
    int rows = m1->getRows();
    int columns = m1->getColumns();

    // Matrix *summed = new Matrix(rows, columns);

    for (int row = 0; row < rows; row++)
    {
        for (int column = 0; column < columns; column++)
        {
            // summed->getMatrix()[row][column] = m1->getMatrix()[row][column] - (m2->getMatrix()[row][column] * multiplicator);
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

static double accuracy(Matrix *AL, Matrix *Y)
{
    Matrix *transponseAL = AL->T();
    Matrix *transponseY = Y->T();
    Matrix *AL_squeeze = squeeze(transponseAL, "max");
    Matrix *Y_squeeze = squeeze(transponseY, "category");

    // AL->printParams("AL");
    // Y->printParams("Y");

    double **al = AL_squeeze->getMatrix();
    double **y = Y_squeeze->getMatrix();

    // AL_squeeze->print("AL_squeeze");
    // Y_squeeze->print("Y_squeeze");

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