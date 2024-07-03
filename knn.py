import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.datasets import fetch_openml
import numpy.random
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def knn(train_images, train_labels, query_image, k):

# Calculate Euclidean distances
  distances = [(i,np.linalg.norm(train_images[i] - query_image),train_labels[i]) for i in range(len(train_images))]

  # Sort distances in ascending order
  sorted_distances = sorted(distances, key=lambda x: x[1])

  # Select the k nearest neighbors
  k_nearest_neighbors = sorted_distances[:k]

  # Count labels freqency
  label_freq = {}
  for i in range(k):
    label = k_nearest_neighbors[i][2]
    if label in label_freq:
      label_freq[label] += 1
    else:
      label_freq[label] = 1

  # Get the most frequent label
  most_frequent_label = max(label_freq, key=label_freq.get)

  return most_frequent_label

def run_knn(n=1000,k=10,prnt=False):

  #initialize
  correct_classifications_cnt = 0
  for i in range(len(test)):
    pred = knn(train[:n],train_labels[:n],test[i],k)
    if pred == test_labels[i]:
      correct_classifications_cnt += 1

  # calculate the accurecy of the predictions
  accuracy = correct_classifications_cnt / len(test)
  if prnt:
    print("The Model Accuracy:", accuracy)

  # Plot predictions accurcy as function of k
  k_values = range(1, 101)
  accuracies = [run_knn(k=k) for k in k_values]
  plt.figure(figsize=(10, 6))
  plt.plot(k_values, accuracies, marker='')
  plt.xlabel('k')
  plt.ylabel('Accuracy')
  plt.title('Accuracy as a function of k')
  plt.grid(True)
  plt.show()

# Plot the prediction accuracy as a function of n=100,200,...,5000 (with k = 1)
n_values = range(100, 5001, 100)
accuracies = [run_knn(n=n,k=1) for n in n_values]
plt.figure(figsize=(10, 6))
plt.plot(n_values, accuracies, marker='')
plt.xlabel('n')
plt.ylabel('Accuracy')
plt.title('Accuracy as a function of n')
plt.grid(True)
plt.show()
