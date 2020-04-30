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


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu")) # add a non-linear property to the neural network. In this way, the network can model more complex relationships and patterns in the data.
model.add(tf.keras.layers.Dense(len(class_names), activation="softmax")) # output neurons  take values between zero and one, so they can represent probability scores.


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_img, train_labels, epochs=6)

prediction = model.predict(test_img)

test_loss, test_acc = model.evaluate(test_img, test_labels)

print(test_acc, test_loss)

for i in range(5):
    plt.imshow(test_img[i])
    plt.title("Actual: " + str(class_names[test_labels[i]]))
    plt.suptitle("Predicition: "+ class_names[np.argmax(prediction[i])])
    plt.show()

