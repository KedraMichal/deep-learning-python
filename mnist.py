import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

np.set_printoptions(suppress=True)

(x_train, y_train), (x_test,y_test) = mnist.load_data()# 60k-10k
#print(x_train[0])# 784 value are the pixels of the 28x28 image, numbers from 0 to 255 (8bits)
#print(y_train[0]) # digit which is on the image

#plt.imshow(x_train[0])
#plt.show()


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss ='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


model.save("number_reader")
new_model = tf.keras.models.load_model('number_reader')


predictions = new_model.predict(x_test)

print(predictions.shape)

print(predictions[0,:])
print(np.argmax(predictions[0]))
acc = 0
for i in range(10000):
    pred = np.argmax(predictions[i,:])
    if pred == y_test[i]:
        acc +=1
    print(pred, y_test[i])


print(acc/10000)
