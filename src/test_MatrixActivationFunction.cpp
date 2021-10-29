#include "./matrix/Matrix.cpp"
#include "./operation/MatrixOperation.cpp"

int main() {
    int rows = 3;
    int cols = 2;
    int iter = 0;

    double **_matrix = new double*[rows];
    for (int row = 0; row < rows; row++) {
        _matrix[row] = new double[cols];
        for (int column = 0; column < cols; column++) {
            _matrix[row][column] = iter++;
        }
    }

    Matrix matrix1 = Matrix(rows, cols, _matrix);
    matrix1.print();

    cout << "Sigmoid" << endl;
    (*sigmoid(matrix1)).print();
    // [[0.5        0.73105858]
    // [0.88079708 0.95257413]
    // [0.98201379 0.99330715]]
    cout << "sigmoidDerivative" << endl;
    (*sigmoidDerivative(matrix1)).print();
    // [[0.25      0.19661193]
    // [0.10499359 0.04517666]
    // [0.01766271 0.00664806]]
    cout << "softmax" << endl;
    (*softmax(matrix1)).print();
    // [[0.26894142 0.73105858]
    // [0.26894142 0.73105858]
    // [0.26894142 0.73105858]]
    cout << "softmaxDerivation" << endl;
    (*softmaxDerivation(matrix1)).print();
    // [[0.19661193 0.19661193]
    // [0.19661193 0.19661193]
    // [0.19661193 0.19661193]]
    cout << "reLu" << endl;
    (*reLu(matrix1)).print();
    // [[0. 1.]
    // [2. 3.]
    // [4. 5.]]
    cout << "reLuDerivation" << endl;
    (*reLuDerivation(matrix1)).print();
    // [[0. 1.]
    // [1. 1.]
    // [1. 1.]]

    return 0;
}