import tensorflow as tf
import keras
import numpy as np


data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])

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



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(decode_opinion(test_data[0], reversed_wd))
