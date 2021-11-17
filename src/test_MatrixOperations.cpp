#include "./matrix/Matrix.cpp"
#include "./operation/MatrixOperation.cpp"

/**
 * @brief Tests for MatrixOperation class
 * 
 * @return int 
 */
int main() {
    // std::cout << "hello wrld! juice world is dead Sad UwU." << "\n";
    std::cout << "Testing Matrix" << "\n";

    int rows = 4;
    int cols = 2;
    int iter = 0;

    double **_matrix = new double*[rows];
    for (int row = 0; row < rows; row++) {
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
    
    cout << "Matrices were not summed, the function returned: ";
    cout << (sum(matrix1, matrix2)) << endl;
    (*sum(matrix2, matrix3)).print();

    (*sumDimension(matrix1)).print();
    (*sumDimension(matrix2)).print();

    (*dot(matrix1, matrix2)).print();
    (*dot(matrix2, matrix1)).print();    
    cout << (dot(matrix1, matrix1)) << endl; 

    (*multiply(matrix2, matrix2)).print();   
    cout << (multiply(matrix1, matrix2)) << endl;

    (*log(matrix1)).print();
    (*log(matrix2)).print();
    
    return 0;
}