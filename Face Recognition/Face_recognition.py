import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def show_orignal_images(pixels):
	#Displaying Orignal Images
	fig, axes = plt.subplots(6, 10, figsize=(11, 7),subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
	plt.show()

def show_eigenfaces(pca):
	#Displaying Eigenfaces
	fig, axes = plt.subplots(3, 8, figsize=(9, 4),subplot_kw={'xticks':[], 'yticks':[]})
	for i, ax in enumerate(axes.flat):
	    ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
	    ax.set_title("PC " + str(i+1))
	plt.show()



#OpenFile
df = pd.read_csv('face_data.txt')
labels = df["target"]
pixels = df.drop(["target"], axis=1)
#SplitData
x_train, x_test, y_train, y_test = train_test_split(pixels, labels)
#PCA
pca = PCA(n_components=160).fit(x_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid(True)
plt.show()

x_train_pca = pca.transform(x_train)

clf = SVC(kernel='rbf', C=1000, gamma=0.01)
clf.fit(x_train_pca, y_train)

x_test_pca = pca.transform(x_test)
y_pred = clf.predict(x_test_pca)
print(classification_report(y_test, y_pred))