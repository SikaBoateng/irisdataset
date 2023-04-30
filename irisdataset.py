from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris= datasets.load_iris()

X = iris.data
y = iris.target

# print(X, y)
# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2)

# print(X_train.shape)
# # should output(120,4) 120 for the number of instances of the training data thus number of rows and 4 for the number of features(features for training)
# print(X_test.shape)
# #output(30,4)(features for testing)
# print(y_train.shape)
# #output(120,) for the number of instances in the label aspect of the train data(labels for training)
# print(y_test.shape)
# #30(labels for testing)


# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)