import tensorflow as tf
import keras
import numpy as np

data = keras.datasets.imdb
np.set_printoptions(suppress=True)

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=100000)
print(train_data.shape)
print(test_data.shape)

word_index = data.get_word_index()
print(word_index)
word_index = {k: (v + 3) for k, v in word_index.items()}
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
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=0, maxlen=200, padding="post")
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0, maxlen=200, padding="post")

print(decode_opinion(test_data[0], reversed_wd))

if keras.models.load_model("model.txt") is None:
    model = keras.Sequential()
    model.add(keras.layers.Embedding(100000, 16))
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

titanic_review = """Titanic is one of my all time favourite films. I'm a 24 year old guy who has probably not cried 
in front of anyone since I was a kid, but this movie plays on my emotions more than almost any other. We all know the 
ship will sink but the build-up to this event - the smugness and over- confidence, assurances of it being 
"unsinkable" - tease the audience, almost to the point where you're like 'is it really going to sink?' I thought the 
performances of everyone were magnificent: I cared about the characters and their lives on and beyond the ship. 
Taking the time to introduce everyone and weave their stories together was masterful, even beyond the main 
characters, like Fabrizio and Murdoch the First Officer and countless more. It was heartbreaking towards the end when 
you see the panic gradually set in and it slowly descends into total chaos. You remember Jack telling Rose all the 
places he is going to take her to and you imagine their love- filled lives together. The Irish mother telling her 
children the fairytale, resigned to their fate. The violinists trying to maintain a spirit of composure amidst the 
carnage. And the music! Wow. I cried for all of this. And especially right at the end when you see the photos of Rose 
is life, knowing she is fulfilled her promise and survived, living a full life, probably never having met anyone she 
loved as much as Jack."""


def encode_review(review):
    review = review.replace("\n", "").replace(".", "").replace(",", "")
    review = review.split(" ")

    review = [i.lower() for i in review]

    encoded_sent = [1]
    for i in review:
        if i in word_index:
            encoded_sent.append(word_index[i])
        else:
            encoded_sent.append(2)  # unknown words

    encoded_sent = np.array([encoded_sent])
    encoded_sent = keras.preprocessing.sequence.pad_sequences(encoded_sent, value=0, maxlen=200, padding="post")

    return encoded_sent


titanic_predict = model.predict(encode_review(titanic_review))
print(titanic_predict)