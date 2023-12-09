import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tsetlinMachine import TsetlinMachine
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

# Load the data from the UCI URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
col_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
             'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
             'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
             'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
             'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
df = pd.read_csv(url, header=None, names=col_names)

# Instantiate the encoders
label_enc = LabelEncoder()
one_hot_enc = OneHotEncoder()

# Encode target variable
df['class'] = label_enc.fit_transform(df['class'])

# One-hot encode the categorical features
df_encoded = pd.get_dummies(df)

# Separate features and target
X = df_encoded.drop('class', axis=1).values
y = df_encoded['class'].values

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarize the dataset with one-hot encoding method (Already done in the previous code)

# Initialize your TsetlinMachine
tm = TsetlinMachine(number_of_clauses=50, number_of_features=X_train.shape[1], number_of_states=300, s=5, threshold=15)

# Train the Tsetlin Machine
epochs = 100
for epoch in range(epochs):
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    tm.fit(X_train, y_train, number_of_examples=X_train.shape[0], epochs=1)

# Predict on the training and test set
y_train_pred = np.array([tm.predict(x) for x in X_train])
y_test_pred = np.array([tm.predict(x) for x in X_test])

# Calculate evaluation metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_prec = precision_score(y_train, y_train_pred)
test_prec = precision_score(y_test, y_test_pred)
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

# Print evaluation metrics
print(f'Training accuracy: {train_acc:.2f}, Precision: {train_prec:.2f}, Recall: {train_recall:.2f}, F1-score: {train_f1:.2f}')
print(f'Test accuracy: {test_acc:.2f}, Precision: {test_prec:.2f}, Recall: {test_recall:.2f}, F1-score: {test_f1:.2f}')

# Plot the training and test accuracy curves
train_accuracy_curve = np.zeros(epochs)
test_accuracy_curve = np.zeros(epochs)

for epoch in range(epochs):
    tm.fit(X_train, y_train, number_of_examples=X_train.shape[0], epochs=1)
    y_train_pred = np.array([tm.predict(x) for x in X_train])
    y_test_pred = np.array([tm.predict(x) for x in X_test])
    train_accuracy_curve[epoch] = accuracy_score(y_train, y_train_pred)
    test_accuracy_curve[epoch] = accuracy_score(y_test, y_test_pred)

plt.figure(figsize=(10, 6))
plt.plot(train_accuracy_curve, label='Train Accuracy')
plt.plot(test_accuracy_curve, label='Test Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
