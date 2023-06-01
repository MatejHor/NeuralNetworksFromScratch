# Neural network from Scratch
Implementation of a feed-forward neural network in C/C++.
Trained it on a Fashion-MNIST dataset using a backpropagation algorithm.

## Requirements
Solution met ALL of the following requirements:
1. Your solution must be compilable and runnable on the AISA server.
2. Your solution achieves at least 88% accuracy.
3. Your solution must finish within 30 minutes. 
   (parse inputs, train, evaluate, export results.)
3. Your solution must contain a runnable script called "RUN" (not run, not 
   RUN.sh, not RUN.exe etc) which compiles, executes and exports the results
   into a files.
4. Your solution must output two files:
    - "trainPredictions" - network predictions for training input vectors 
    - "actualTestPredictions" - network predictions for testing input vectors
   The format of these files is the same as the supplied training/testing
   labels: 
    - One prediction per line.
    - Prediction for i-th input vector (ordered by the input .csv file) must
      be on i-th line in the associated output file.
    - Each prediction is a single integer 0 - 9.

## Dataset
Fashion MNIST (https://arxiv.org/pdf/1708.07747.pdf) a modern version of a
well-known MNIST (http://yann.lecun.com/exdb/mnist/). It is a dataset of
Zalando's article images ‒ consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image, 
associated with a label from 10 classes. The dataset is in CSV format. There 
are four data files included:
    - fashion_mnist_train_vectors.csv   - training input vectors
    - fashion_mnist_test_vectors.csv    - testing input vectors
    - fashion_mnist_train_labels.csv    - training labels
    - fashion_mnist_test_labels.csv     - testing labels
   
## Contributors
* Matej Horniak
* Lukas Mikula


