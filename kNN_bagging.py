# import all necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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
for i in range(1, 31):
    for j in range(10, 31):
        bagknn = BaggingClassifier(KNeighborsClassifier(n_neighbors=i), n_estimators=j)
        bagknn.fit(X_train, y_train)
        pred_ij = bagknn.predict(X_test)
        error.append(np.mean(pred_ij != y_test))
        print('k value =', i, '| #(bootstrap samples) =', j, '--> % of error =', error[i + j - 11])

