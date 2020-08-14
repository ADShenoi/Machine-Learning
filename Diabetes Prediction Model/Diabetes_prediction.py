import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Read the csv file
data =pd.read_csv('pima-data.txt')

# Change the boolean values into 0's and 1's
diabetes_map = {True: 1, False: 0}
data['diabetes'] = data['diabetes'].map(diabetes_map)

# Plot the correlation of each feature with respect to the Target Output
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

# Plot the heat map
g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

# Defining the Dependent and Independent feature
x = data.drop('diabetes', axis=1).values
y = data['diabetes'].values

# Split the Dataset into Training set and Testing set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=10)

# Fill the missing spots with mean
fill_values = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)

# Train the model
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, Y_train.ravel())

predict_train_data = random_forest_model.predict(X_test)

# Print the Accuracy of the model
print("Accuracy of the model = {0:3f}".format(metrics.accuracy_score(Y_test, predict_train_data)))