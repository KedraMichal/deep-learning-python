import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
import tensorflow
import keras

np.set_printoptions(suppress=True)

data = pd.read_csv("matches.csv", sep=";",  decimal=",")

# place 0-home, 1-away, result 2-win 1-draw 0-lose
data = data[["place", "value ratio", "result"]]

data = np.asarray(data)

x = data[:, 0:2]
y = data[:, 2]


x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size= 0.2, random_state=0)


model = keras.Sequential()
model.add(keras.layers.Dense(2))
model.add(keras.layers.Dense(32,activation="relu"))
model.add(keras.layers.Dense(3, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=20)

acc = model.evaluate(x_test, y_test)
print(model.predict(x_test), y_test)

