import tensorflow as tf
import keras
import numpy as np


data = keras.datasets.imdb
np.set_printoptions(suppress=True)

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
print(train_data.shape)
print(test_data.shape)

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


reversed_wd = [[key, value] for (value, key) in word_index.items()]
reversed_wd = dict(reversed_wd)

def decode_opinion(opinion, coding):
    decoded = []
    for i in opinion:
        decoded.append(coding.get(i))

    return " ".join(decoded)


def reviev_stats(lists):
    words_used = np.array([])
    for i in lists:
        words_used = np.append(words_used, len(i))

    return np.min(words_used), np.max(words_used), np.mean(words_used)


print(reviev_stats(test_data))
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=0, maxlen= 200, padding="post")
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0, maxlen= 200, padding="post")

print(decode_opinion(test_data[0], reversed_wd))

if keras.models.load_model("model.txt") is None:
    model = keras.Sequential()
    model.add(keras.layers.Embedding(10000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_data, train_labels, epochs=10)

    model.save("model.txt")

model = keras.models.load_model("model.txt")

results = model.evaluate(test_data, test_labels)
print(results)


for i in range(10):
    predict = model.predict(test_data)
    print(predict[i])
    print(decode_opinion(test_data[i], reversed_wd))



