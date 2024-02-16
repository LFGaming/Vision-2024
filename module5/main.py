import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

digits = datasets.load_digits()
random_state = random.randint(0, 100)
# Splitting the data into approximately two-thirds for training and one-third for testing
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42) # choose what state you're in
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=random_state)


print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X_train, y_train)

# Predicting on a sample test data
print("Prediction:", clf.predict(X_test[-4:-3]))

plt.imshow(X_test[-4].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')

# Train the SVM classifier again
clf.fit(X_train, y_train)

# Predict the labels for the training data
train_predictions = clf.predict(X_train)

# Calculate the accuracy of the model on the training data
train_accuracy = accuracy_score(y_train, train_predictions)

print("Training Accuracy:", train_accuracy)

# Predict the labels for the test data
test_predictions = clf.predict(X_test)

# Calculate the accuracy of the model on the test data
test_accuracy = accuracy_score(y_test, test_predictions)

print("Test Accuracy:", test_accuracy)

plt.show()
