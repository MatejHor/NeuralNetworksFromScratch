#include "matrix/Matrix.cpp"

int main() {
    std::cout << "hello wrld! juice world is dead Sad UwU." << "\n";

    int rows = 4;
    int cols = 2;
    
    Matrix matrix = Matrix(rows, cols);
    int iter = 0;

    double **_matrix = new double*[rows];
    for (int row = 0; row < rows; row++)
    {
        _matrix[row] = new double[cols];
        for (int column = 0; column < cols; column++) {
            _matrix[row][column] = iter++;
        }
    }

    Matrix matrix1 = Matrix(rows, cols, _matrix);

    Matrix matrix2 = Matrix(matrix1);

    cout << "Matrices are the same: " << matrix1.operator==(matrix2) << "\n";

    // matrix = matrix1;

    // (matrix1.T()).print();
    
    // cout << "KOKOT DEBUGGING" << endl;
    matrix1.print();

    matrix1 = *matrix1.T();
    // (matrix1.T())->print();

    matrix1.print();

    matrix2.print();

    // matrix.print();

    cout << "Matrices are the same: " << matrix1.operator==(matrix2) << "\n";
    cout << "Matrices are not the same: " << matrix1.operator!=(matrix2) << "\n";
    
    return 0;
}