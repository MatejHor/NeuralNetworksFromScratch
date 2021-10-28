#include "./matrix/Matrix.cpp"
#include "./dataset/Dataset.cpp"

#include <iostream>
#include <chrono>
#include <random>

int main()
{
    std::string xFileName = "../data/fashion_mnist_sample_vectors.csv";
    std::string yFileName = "../data/fashion_mnist_sample_labels.csv";

    Dataset datenkaplsky = Dataset(xFileName, yFileName);
    // datenkaplsky.getX()->T()->print();
    
    datenkaplsky.print(5);
}