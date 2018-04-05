from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from jupyterthemes import jtplot
jtplot.style()


###  Load dataset
file = "train.csv"
ds = pd.read_csv(file)
data = ds.values

y = data[:, 0]
X = data[:, 1:]


# Split the data into 30% test and 70% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



#Euclidean Distance
def dist(p1,p2):
	return np.sqrt(((p2-p1)**2).sum())


# calculate value of k i.e. square root of number of samples in your training dataset
K = math.floor(math.sqrt(X_train.shape[0]))
K = int(K)
if K%2 == 0 :
	K += 1


# K-nearest Neighbor Algorithm from sratch
def knn(trainA,labelA,query_point, k) :

	val = []

	for i in range(trainA.shape[0]) :
		v = [ dist(query_point,trainA[i,:]), labelA[i] ]
		val.append(v)

	#sort the array acc to distances
	val = sorted(val)

	# pick top k nearest distances
	predArr = np.array(val[:k])

	# find frequency of each label and print the label whose frequency is more
	newPredArr = np.unique(predArr[:,1],return_counts = True)

	index = newPredArr[1].argmax()

	return newPredArr[0][index]



# we multiply random number b/w 0 & 1 with 25200 and return answer in integer
random_index = int(np.random.random() * X_train.shape[0])
query_point = X_train[random_index]


# calling knn function
res = knn(X_train, y_train, query_point, K)

plt.figure(0)
plt.imshow(query_point.reshape((28, 28)), cmap='gray')
plt.show()

print "Our predicted number ", int(res), " matches with actual number ", y_train[random_index], " as well as with the image."


#########   Calculating accuracy of various classification algorithms

clf = GaussianNB()
clf.fit(X_train,y_train)
print "accuracy of Naive Bayes is : ", clf.score(X_test,y_test)

clf = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split = 40)
clf.fit(X_train,y_train)
print "accuracy of Decision Tree is : ", clf.score(X_test,y_test)

clf = RandomForestClassifier(criterion='gini', n_estimators = 100, random_state=0, n_jobs=-1)
clf.fit(X_train,y_train)
print "accuracy of Random Forest is : ", clf.score(X_test,y_test)
