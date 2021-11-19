#include "Dataset.h"

/**
 * @brief Construct a new Dataset:: Dataset object
 * 
 * @param xFileName file name/path of the file with vectors
 * @param yFileName file name/path of the file with labels
 * @param maxRow number of rows to read from the files
 * @param verbose flag for printing the datasets, for debugging purposes etc.
 */
Dataset::Dataset(string xFileName, string yFileName, int maxRow, bool verbose): maxRow(maxRow)
{
    int rows = (maxRow != -1) ? maxRow : getRows(xFileName);
    int columns = getColumns(xFileName);

    double **xData = readData(xFileName, rows, columns, true);
    X = Matrix(rows, columns, xData).T();

    double **yData = readData(yFileName, rows, 10, false);
    Y = Matrix(rows, 10, yData).T();

    if (verbose)
    {
        X->printParams("X:");
        Y->printParams("Y:");
    }
}

/**
 * @brief iterate throw whole file and gets the number of rows in this specific file
 * 
 * @param file file name/path of the file
 * @return int number of rows in the file
 */
int Dataset::getRows(string file)
{
    ifstream my_file(file);
    string myText;

    int count_rows = 0;

    while (getline(my_file, myText))
    {
        count_rows++;
    }

    my_file.close();

    return count_rows;
}

/**
 * @brief gets number of columns in the file
 * 
 * @param file file name/path of the file
 * @return int number of columns in the file
 */
int Dataset::getColumns(string file)
{
    ifstream my_file(file);
    string myText;

    getline(my_file, myText);
    stringstream one_line(myText);
    string value;

    int count_values = 0;
    while (getline(one_line, value, ','))
    {
        count_values++;
    }

    my_file.close();

    return count_values;
}

/**
 * @brief reads data from given file, used for loading datasets
 * 
 * @param file file name/path of the file
 * @param rows number of rows to read
 * @param columns number of columns to read
 * @param X binary flag, when true, vectors are read, when false, labels
 * @return double** 2D array representation of matrix with dataset data
 */
double **Dataset::readData(string file, int rows, int columns, bool X)
{
    ifstream read_vectors(file);
    string myText;

    double **_matrix = new double *[rows];
    for (int row = 0; row < rows; row++)
    {
        _matrix[row] = new double[columns];
    }

    int i = 0;
    int j = 0;
    while (getline(read_vectors, myText))
    {
        stringstream one_line(myText);
        string value;
        j = 0;
        if (X)
        {
            while (getline(one_line, value, ','))
            {
                _matrix[i][j++] = stod(value) / 255.0;
            }
        }
        else
        {
            for (int j = 0; j < columns; j++)
            {
                _matrix[i][j] = 0;
            }
            _matrix[i][stoi(myText)] = 1;
        }
        i++;
        if (i == rows)
            break;
    }

    return _matrix;
}

/**
 * @brief prints the dataset object, the number of printed rows must be specified
 * 
 * @param limit number of printed rows of the dataset object
 */
void Dataset::print(int limit)
{
    cout << "Dataset x rows=" << X->getRows() << ", columns=" << X->getColumns() << ", &=" << (this) << ")" << endl;

    if (limit > X->getRows())
    {
        limit = X->getRows();
    }

    for (int i = 0; i < limit; i++)
    {
        for (int j = 0; j < X->getColumns(); j++)
        {
            cout << X->getMatrix()[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Dataset Y rows=" << Y->getRows() << ", columns=" << Y->getColumns() << ", &=" << (this) << ")" << endl;

    if (limit > Y->getRows())
    {
        limit = Y->getRows();
    }

    for (int i = 0; i < limit; i++)
    {
        for (int j = 0; j < Y->getColumns(); j++)
        {
            cout << Y->getMatrix()[i][j] << " ";
        }
        cout << endl;
    }
}

// double Dataset::f1_mikro(Matrix *AL)
// {
//     Matrix *yMatrix = squeeze(getY(), "category");
//     double **y = yMatrix->getMatrix();
//     double **al = AL->getMatrix();

//     double TP = 0;
//     double TN = 0;
//     double FP = 0;
//     double FN = 0;

//     for (int i = 0; i < 10; i++)
//     {
//         for (int row = 0; row < Y->getRows(); row++)
//         {
//             (al[row][0] == i && y[row][0] == i && ++TP);
//             (al[row][0] != i && y[row][0] != i && ++TN);
//             (al[row][0] == i && y[row][0] != i && ++FP);
//             (al[row][0] != i && y[row][0] == i && ++FN);
//         }
//     }

//     double precision = TP / (TP + FP);
//     double recall = TP / (TP + FN);
//     yMatrix->~Matrix();
//     return 2 * ((precision * recall) / (precision + recall));
// }


