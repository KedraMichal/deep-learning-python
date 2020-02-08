import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

###  Data set contains 60k traning images of numbers from 0 to 9
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])  # one x contains 784 values (number of pixels - 28x28 image), from 0 to 255 (8bits)
print(y_train[0])  # our labels, digit which is on the image from 0 to 9

plt.imshow(x_train[0])
plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save("number_reader")
new_model = tf.keras.models.load_model('number_reader')

predictions = new_model.predict(x_test)

print(predictions.shape)

acc = 0
for i in range(10000):
    pred = np.argmax(predictions[i, :])
    if pred == y_test[i]:
        acc += 1
    print(pred, y_test[i])

print(acc / 10000)
