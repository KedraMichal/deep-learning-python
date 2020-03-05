import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = tf.keras.datasets.fashion_mnist

(train_img, train_labels),(test_img, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#plt.imshow(train_img[1], cmap="binary")
#print(train_labels[1])
#plt.show()

train_img = tf.keras.utils.normalize(train_img, axis=1)
test_img = tf.keras.utils.normalize(test_img, axis =1)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(class_names), activation="softmax")])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_img, train_labels, epochs=4)

prediction = model.predict(test_img)

