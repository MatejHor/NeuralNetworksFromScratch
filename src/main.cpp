#include "./matrix/Matrix.cpp"
#include "./operation/MatrixOperation.h"

int main() {
    // std::cout << "hello wrld! juice world is dead Sad UwU." << "\n";
    std::cout << "Testing Matrix" << "\n";

    int rows = 4;
    int cols = 2;
    int iter = 0;

    double **_matrix = new double*[rows];
    for (int row = 0; row < rows; row++)
    {
        _matrix[row] = new double[cols];
        for (int column = 0; column < cols; column++) {
            _matrix[row][column] = iter++;
        }
    }

    Matrix matrix1 = Matrix(3, 3);
    Matrix matrix2 = Matrix(rows, cols, _matrix);
    Matrix matrix3 = Matrix(matrix2);

    matrix2.print();
    matrix1.print();
    matrix1 = *matrix2.T();
    matrix1.print();
    matrix2.print();
    matrix3.print();

    cout << "Matrices (" << &matrix2 << "," << &matrix3 << ") are the same: " << ((matrix2 == matrix3) ? "true" : "false") << endl;
    cout << "Matrices (" << &matrix1 << "," << &matrix3 << ") are the same: " << ((matrix1 == matrix3) ? "true" : "false") << endl;
    cout << "Matrices (" << &matrix1 << "," << &matrix2 << ") are the same: " << ((matrix1 == matrix2) ? "true" : "false") << "\n";
    cout << "Matrices (" << &matrix2 << "," << &matrix3 << ") are not the same: " << ((matrix2 != matrix3) ? "true" : "false") << "\n";
    cout << "Matrices (" << &matrix1 << "," << &matrix3 << ") are not the same: " << ((matrix1 != matrix3) ? "true" : "false") << "\n";
    return 0;
}