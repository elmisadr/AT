import numpy as np
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tsetlinMachine import TsetlinMachine

# Load the Iris dataset from UCI URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
response = requests.get(url).content.decode('utf-8')
df = pd.read_csv(StringIO(response), header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Binarize the dataset using the threshold method
threshold = 0.5
X_binarized = np.where(X >= threshold, 1, 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_binarized, y, test_size=0.2, random_state=42)

# Tsetlin Machine parameters
number_of_clauses = 800
number_of_features = X_train.shape[1]
number_of_states = 100
s = 4
threshold = 1
# Tsetlin Machine model
tm = TsetlinMachine(number_of_clauses, number_of_features, number_of_states, s, threshold)
# Training the Tsetlin Machine
epochs = 100
train_accuracy_list = []
test_accuracy_list = []

for epoch in range(epochs):
    for example_id in range(X_train.shape[0]):
        target_class = int(y_train[example_id])
        tm.update(X_train[example_id], target_class)

    # Evaluate on training and testing sets
    train_accuracy = tm.evaluate(X_train, y_train, X_train.shape[0])
    test_accuracy = tm.evaluate(X_test, y_test, X_test.shape[0])
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)

# Calculate additional metrics
y_pred_train = [tm.predict(x) for x in X_train]
y_pred_test = [tm.predict(x) for x in X_test]

precision_train = precision_score(y_train, y_pred_train, average='macro')
recall_train = recall_score(y_train, y_pred_train, average='macro')
f1_train = f1_score(y_train, y_pred_train, average='macro')

precision_test = precision_score(y_test, y_pred_test, average='macro')
recall_test = recall_score(y_test, y_pred_test, average='macro')
f1_test = f1_score(y_test, y_pred_test, average='macro')

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training Precision:", precision_train)
print("Testing Precision:", precision_test)
print("Training Recall:", recall_train)
print("Testing Recall:", recall_test)
print("Training F1 Score:", f1_train)
print("Testing F1 Score:", f1_test)

# Plot train and test accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), train_accuracy_list, label="Train Accuracy")
plt.plot(range(epochs), test_accuracy_list, label="Test Accuracy")
plt.title('Train and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
