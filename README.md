## README

### Handwritten Digit Recognition Using KNN

This project implements a handwritten digit recognition model using the k-Nearest Neighbors (KNN) algorithm. The model is trained and tested on the MNIST dataset, which consists of 70,000 images of handwritten digits.

### Description

The provided code uses the MNIST dataset for training and testing the KNN model. The dataset contains 70,000 28x28 pixel images of handwritten digits. The data is split into training and test sets, with 10,000 images used for training and 1,000 images for testing.

### Implementation

1. **Data Preparation**:
   - The dataset is loaded and split into training and test sets using random sampling.
   
2. **KNN Algorithm**:
   - The `knn` function calculates the Euclidean distance between the query image and all training images.
   - It selects the `k` nearest neighbors and determines the most frequent label among them.
   
3. **Model Evaluation**:
   - The `run_knn` function evaluates the model accuracy by predicting the labels for the test set and calculating the percentage of correct predictions.
   - It plots the prediction accuracy as a function of `k` (number of neighbors) and `n` (number of training samples).

### Results

- The model's accuracy is evaluated with different values of `k` and `n`.
- With `k = 1` and increasing `n`, the accuracy improves but not linearly. This indicates a limitation in the pairing method for the given problem.

### Usage

Run the code to see the accuracy of the KNN model with different parameters. Modify the values of `k` and `n` to observe their effect on the model's performance.

### Dependencies

- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`

### Statistical Analysis

The project also includes a statistical analysis of the model's accuracy. A completely random predictor would achieve an accuracy of around 10%. The implemented KNN model significantly outperforms this, achieving an accuracy of approximately 85.8% with `k=10` and `n=1000`.

### Conclusion

This project demonstrates the implementation of a KNN-based handwritten digit recognition model using the MNIST dataset. It highlights the impact of different parameters on the model's accuracy and provides a visual representation of these effects.

### Author

Ibraheem Abomokh

For more detailed information, refer to the provided statistical analysis document.
