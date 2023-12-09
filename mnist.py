import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tsetlinMachine import TsetlinMachine
from skimage.filters import threshold_otsu
import random
import math
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Binarize the dataset with Otsu's method
def otsu_binarization(images):
    binarized_images = []
    for image in images:
        thresh = threshold_otsu(image)
        binarized_image = image > thresh
        binarized_images.append(binarized_image)
    return np.array(binarized_images).astype(np.int32)

# Binarize training and test sets
X_train_binarized = otsu_binarization(X_train)
X_test_binarized = otsu_binarization(X_test)

# Flatten the binary images into one-dimensional feature vectors
X_train_flattened = X_train_binarized.reshape(X_train_binarized.shape[0], -1)
X_test_flattened = X_test_binarized.reshape(X_test_binarized.shape[0], -1)

# Split the dataset into 80% training and 20% test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_flattened, y_train, test_size=0.2, random_state=42)

# Initialize your TsetlinMachine
tm = TsetlinMachine(number_of_clauses=8000, number_of_features=X_train.shape[1], number_of_states=256, s=5, threshold=800)
alpha = tm.calculate_equation(s, decay_rate, epoch)
# Train the Tsetlin Machine
epochs = 500
for epoch in range(epochs):
    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    for example_id in range(X_train.shape[0]):
        target_class = int(y_train[example_id])
        tm.update(X_train[example_id], target_class)

# Predict on the training and test set
y_train_pred = np.array([tm.predict(x) for x in X_train])
y_test_pred = np.array([tm.predict(x) for x in X_test])

# Calculate training and test accuracies
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Calculate training and test precision
train_prec = precision_score(y_train, y_train_pred, average='weighted')
test_prec = precision_score(y_test, y_test_pred, average='weighted')

# Calculate training and test recall
train_recall = recall_score(y_train, y_train_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

# Print training and test accuracies, precision, and recall
print(f'Training accuracy: {train_acc:.2f}, Precision: {train_prec:.2f}, Recall: {train_recall:.2f}')
print(f'Test accuracy: {test_acc:.2f}, Precision: {test_prec:.2f}, Recall: {test_recall:.2f}')

# Plot the training and test accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

       
