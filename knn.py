import numpy as np
import matplotlib.pyplot as plt

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