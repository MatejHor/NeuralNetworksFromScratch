#include "./matrix/Matrix.cpp"
#include "./dataset/Dataset.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./nn/NeuralNetwork.cpp"

int main()
{
    double layer_dims[] = {784, 255, 225, 200, 10};
    NeuralNetwork model = NeuralNetwork(
        layer_dims, //layer_dims
        10, //epochs
        16, //batchSize
        0.001, //learning_rate
        true //verbose
    );
}