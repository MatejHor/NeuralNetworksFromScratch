#include "./matrix/Matrix.cpp"
#include "./dataset/Dataset.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./nn/NeuralNetwork.cpp"

int main()
{
    bool VERBOSE = true;

    auto start = high_resolution_clock::now();
    Dataset train = Dataset(
        "../data/fashion_mnist_train_vectors.csv",
        "../data/fashion_mnist_train_labels.csv",
        40000, // size of train dataset
        VERBOSE);
    auto stop = high_resolution_clock::now();
    cout << "Train dataset load time: " << duration_cast<seconds>(stop - start).count() << " seconds" << endl;

    start = high_resolution_clock::now();
    Dataset test = Dataset(
        "../data/fashion_mnist_test_vectors.csv",
        "../data/fashion_mnist_test_labels.csv",
        10000, // size of test dataset
        VERBOSE);
    stop = high_resolution_clock::now();
    cout << "Test dataset load time: " << duration_cast<seconds>(stop - start).count() << " seconds" << endl;
    
    NeuralNetwork model = NeuralNetwork(
        10, //epochs
        256, //batchSize
        4.0, //learning_rate
        0.9 //beta
    );

    start = high_resolution_clock::now();
    model.fit(&train);
    stop = high_resolution_clock::now();
    cout << "Train time: " << duration_cast<seconds>(stop - start).count() << " seconds" << endl;

    start = high_resolution_clock::now();
    double acc = model.transform(&test);
    stop = high_resolution_clock::now();
    cout << "Test accuracy: " << acc << " time:" << duration_cast<seconds>(stop - start).count() << " seconds" << endl;
    return 0;
}