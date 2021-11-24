#include "./matrix/Matrix.cpp"
#include "./dataset/Dataset.cpp"
#include "./operation/MatrixOperation.cpp"
#include "./nn/NeuralNetworkMultiLayer.cpp"

int main()
{
    bool VERBOSE = true;

    auto start = high_resolution_clock::now();
    // Train dataset with full length (60 000)
    Dataset train = Dataset(
        "./data/fashion_mnist_train_vectors.csv",
        "./data/fashion_mnist_train_labels.csv",
        60000, // size of train dataset
        VERBOSE);
    auto stop = high_resolution_clock::now();
    cout << "Train dataset load time: " << duration_cast<seconds>(stop - start).count() << " seconds" << endl;

    start = high_resolution_clock::now();
    // Test dataset with full length (10 000)
    Dataset test = Dataset(
        "./data/fashion_mnist_test_vectors.csv",
        "./data/fashion_mnist_test_labels.csv",
        10000, // size of test dataset
        VERBOSE);
    stop = high_resolution_clock::now();
    cout << "Test dataset load time: " << duration_cast<seconds>(stop - start).count() << " seconds" << endl;

    // Hierarchy of network
    vector<int> layer = {784, 256, 10};
    NeuralNetworkMultiLayer model = NeuralNetworkMultiLayer(
        layer, // layers
        30,    //epochs
        256,   //batchSize
        0.2,   //learning_rate
        0.9    //beta
    );

    // Training of network
    start = high_resolution_clock::now();
    model.fit(&train);
    stop = high_resolution_clock::now();
    cout << "Train time: " << duration_cast<seconds>(stop - start).count() << " seconds (" << duration_cast<minutes>(stop - start).count() << " minutes)" << endl;

    // Predict on train dataset
    start = high_resolution_clock::now();
    double acc = model.transform(&train, "./trainPredictions");
    stop = high_resolution_clock::now();
    cout << "Train accuracy: " << acc << " Time: " << duration_cast<seconds>(stop - start).count() << " seconds" << endl;

    // Predict on test dataset
    start = high_resolution_clock::now();
    acc = model.transform(&test, "./actualTestPredictions");
    stop = high_resolution_clock::now();
    cout << "Test accuracy: " << acc << " Time: " << duration_cast<seconds>(stop - start).count() << " seconds" << endl;
    return 0;
}
