import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Reading the Dataset
df = pd.read_csv('heart_disease.txt')

# Get the correlation of each feature with respect to the Target Output
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

# Plot a Heat map for the same
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

# Converting the categorical values in the Dataset into Dummy variables
dataset = pd.get_dummies(df, columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

# ReScaling the data
standardScalar = StandardScaler()
columns_to_scale = ['age', 'trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale] = standardScalar.fit_transform(dataset[columns_to_scale])

# Defining the Dependent and Independent feature
x = dataset.drop(['target'], axis=1)
y = dataset['target']

# Finding the most suitable k value
knn_score = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_classifier,x,y,cv=10)
    knn_score.append((score.mean()))

# Plot each k value and it's respective score in order to understand and find out the suitable k value
plt.plot([k for k in range(1,21)], knn_score, color = 'red')
for i in range(1,21):
    plt.text(i,knn_score[i-1],(i,knn_score[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbours (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifer scores for different K Values')
plt.rcParams["figure.figsize"] = (20,23)
plt.show()

# Training the model
knn_classifier = KNeighborsClassifier(n_neighbors=12)

# Predicting the Accuracy of the model
score = cross_val_score(knn_classifier,x,y,cv=10)
print('Accuracy = ', score.mean())