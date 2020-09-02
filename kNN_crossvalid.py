# import all necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

# read data
wine = np.loadtxt("wine.txt", delimiter=',', dtype=np.float)

# create training and test samples, and standardize predicates
X = wine[:, 1:13]
y = wine[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# implement the model. count the percentage of errors. display information on the screen
error = []
for i in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    print('k value =',i,'--> % of error =',error[i-1])

# visualizing data
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), error, color='green', linestyle='dashed', marker='o',
         markerfacecolor='yellow', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('% Error')
plt.show()
