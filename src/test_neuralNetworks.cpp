#include "./matrix/Matrix.cpp"
#include "./dataset/Dataset.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./nn/NeuralNetwork.cpp"

int main()
{
    bool VERBOSE = true;
    Dataset train = Dataset(
        "../data/fashion_mnist_train_vectors.csv",
        "../data/fashion_mnist_train_labels.csv",
        3000, // size of train dataset
        VERBOSE);

    Dataset test = Dataset(
        "../data/fashion_mnist_test_vectors.csv",
        "../data/fashion_mnist_test_labels.csv",
        1000, // size of test dataset
        VERBOSE);

    NeuralNetwork model = NeuralNetwork(
        10, //epochs
        256, //batchSize
        4.0, //learning_rate
        0.9 //beta
    );

    model.fit(&train);

    double accuracy = model.transform(&test);
    cout << "Test Accuracy: " << acc << endl;

    model.~NeuralNetwork();
    return 0;
}