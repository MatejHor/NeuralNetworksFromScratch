#include "./matrix/Matrix.cpp"
#include "./dataset/Dataset.cpp"

int test()
{
    std::string xFileName = "../data/fashion_mnist_train_vectors.csv";
    std::string yFileName = "../data/fashion_mnist_train_labels.csv";

    Dataset datenkaplsky = Dataset(xFileName, yFileName);

    datenkaplsky.print(5);
}