#include "Dataset.h"

Dataset::Dataset(string xFileName, string yFileName)
{
    int rows = getRows(xFileName);
    int columns = getColumns(xFileName);

    double** xData = readData(xFileName, rows, columns, true);
    X = Matrix(rows, columns, xData).T();

    double** yData = readData(yFileName, rows, 10, false);
    Y = Matrix(rows, 10, yData).T();

    X->printParams("X:");
    Y->printParams("Y:");
}

int Dataset::getRows(string file)
{
    ifstream my_file(file);
    string myText;

    int count_rows = 0;

    while (getline(my_file, myText))
    {
        // stringstream one_line(myText);
        // string value;
        count_rows++;
        // count_values = 0;
        // while(getline(one_line, value, ',')) {
        //     count_values++;
        // }
    }

    my_file.close();

    return count_rows;
}

int Dataset::getColumns(string file)
{
    ifstream my_file(file);
    string myText;

    getline(my_file, myText);
    stringstream one_line(myText);
    string value;

    int count_values = 0;
    while(getline(one_line, value, ',')) {
        count_values++;
    }

    my_file.close();

    return count_values;
}

double** Dataset::readData(string file, int rows, int columns, bool X)
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
                _matrix[i][j++] = stod(value);
            }
        }
        else
        {
            for (int j= 0; j < columns; j++) {
                _matrix[i][j] = 0;
            }
            _matrix[i][stoi(myText)] = 1;
        }
        i++;
    }

    return _matrix;
}

void Dataset::print(int limit)
{
    cout << "Dataset x rows=" << X->getRows() << ", columns=" << X->getColumns() << ", &=" << (this) << ")" << endl; 

    if (limit > X->getRows()) {
        limit = X->getRows();
    }

    for (int i = 0; i < limit; i++) {
        for (int j = 0; j < X->getColumns(); j++) {
            cout << X->getMatrix()[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Dataset Y rows=" << Y->getRows() << ", columns=" << Y->getColumns() << ", &=" << (this) << ")" << endl; 

    if (limit > Y->getRows()) {
        limit = Y->getRows();
    }

    for (int i = 0; i < limit; i++) {
        for (int j = 0; j < Y->getColumns(); j++) {
            cout << Y->getMatrix()[i][j] << " ";
        }
        cout << endl;
    }
}
