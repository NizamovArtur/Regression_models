# import all necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# read data
wine = np.loadtxt("wine.txt", delimiter=',', dtype=np.float)
with open('names.txt') as file_object:
    lines = file_object.readlines()
preds = []
for line in lines:
    preds.append(line.rstrip())

# create training and test samples, and standardize predicates
X = wine[:, 1:14]
y = wine[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# implement the model. count the percentage of errors. display information on the screen
error = []
for i in range(100,151):
    for j in range(2,6):
        gbc = GradientBoostingClassifier(max_depth=j, n_estimators=i)
        gbc.fit(X_train, y_train)
        pred_ij = gbc.predict(X_test)
        error.append(np.mean(pred_ij != y_test))
        print('#(trees) =', i,'| max depth =', j, '--> % of error =', error[i + j - 102])
