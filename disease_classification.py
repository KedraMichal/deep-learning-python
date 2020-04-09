import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import tensorflow
import keras


data = pd.read_csv("heart_disease_dataset.csv")


# print(data.num.value_counts())
# groupby_class = data.groupby("num")
# print(groupby_class)
data = data.dropna(how="any")
data = data.drop(columns=["ca", "thal", "restecg", "trestbps", "fbs", "chol"])
data = data.to_numpy()

x = data[:, 0:6]
y = data[:, 7]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

pipe = Pipeline([('knn', KNeighborsClassifier())])
parameters = {'knn__n_neighbors': [x for x in range(1, 20)]}
grid = GridSearchCV(pipe, parameters, cv=5)
grid.fit(x, y)
print(grid.best_params_)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
acc = knn.score(x_test, y_test)
print(acc)


model = keras.Sequential()

model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)
acc = model.evaluate(x_test, y_test)

print(acc)