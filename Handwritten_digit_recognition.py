import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']

some_digit = x[36001]
some_digit_image = some_digit.reshape(28,28)   # Reshape to visualize the number

# Plot the Number
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# Split the dataset into training set and testing set
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)  # Shuffle the dataset so that if the dataset is ordered, it will be spread evenly
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train==2)
y_test_2 = (y_test==2)

# Training Model
clf = LogisticRegression(tol = 0.1)
clf.fit(x_train, y_train_2)
print(clf.predict([some_digit]))

# Accuracy Checking
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(a.mean())

