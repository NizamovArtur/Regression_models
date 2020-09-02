# import all necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

# implement regression model
model = LinearRegression().fit(X_train, y_train)
pred_y = model.predict(X_test)

# count the percentage of errors and find the most significant predictor
error = np.mean(np.round(pred_y) != y_test)
mip = np.argmax(np.abs(model.coef_))

# display information on the screen
print('% of error =', error)
print('Most influental predictor =', preds[mip])
