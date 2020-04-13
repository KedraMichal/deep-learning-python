import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import tensorflow
import keras
from keras.wrappers.scikit_learn import KerasClassifier

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

pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
parameters = {'knn__n_neighbors': [x for x in range(1, 20)]}
grid = GridSearchCV(pipe, parameters, cv=10)
grid.fit(x, y)
print(grid.best_params_)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
acc_knn = knn.score(x_test, y_test)


# ### hyperparameters optimization
# def create_model():
#     model = keras.Sequential()
#     model.add(keras.layers.Dense(8, activation='relu'))
#     model.add(keras.layers.Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
#
# model = KerasClassifier(build_fn=create_model, verbose=0)
#
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [100, 200, 500, 750, 1000]
# param_grid = dict(batch_size=batch_size, epochs=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10)
# grid_result = grid.fit(x, y)
#
# ### stats for specified hyperparameters
# print("Best: {} using {}" .format(grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("{} ({}) with: {}" .format(mean, stdev, param))


### model with choosen hyperparameters
model = keras.Sequential()
model.add(keras.layers.Dense(8))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=750, batch_size=20)
acc = model.evaluate(x_test, y_test)

print("KNN accuracy: {}".format(acc_knn))
print("Dl accuracy: {}".format(acc))


