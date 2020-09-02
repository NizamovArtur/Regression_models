# import all necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from matplotlib import pyplot as plt

# read data
wine = np.loadtxt("wine.txt", delimiter=',', dtype=np.float)

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
for i in range(10, 51):
    model = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=i).fit(
        X_train,
        y_train)
    pred_y = model.predict(X_test)
    error.append(np.mean(np.round(pred_y) != y_test))
    print('#(bootstrap samples) =',i,'--> % of error =',error[i-10])


# visualizing data
plt.figure(figsize=(10, 5))
plt.plot(range(10, 51), error, color='green', linestyle='dashed', marker='o',
         markerfacecolor='yellow', markersize=10)
plt.title('Error Rate #(bootstrap samples)')
plt.xlabel('#(bootstrap samples)')
plt.ylabel('% of error')
plt.show()
