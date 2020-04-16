import numpy as np


X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]))
y = np.array(([-1], [1], [-1], [-1]))


def tanh_derivitave(x):
    return 1 - (np.tanh(x)) ** 2


class NeuralNetwork:
    def __init__(self, x, y):
        neurons_in_hidden_layer = 4
        self.input = x
        self.y = y
        self.weights1 = np.random.rand(X.shape[1], neurons_in_hidden_layer)
        self.weights2 = np.random.rand(neurons_in_hidden_layer, 1)

    def feed_forward(self):
        self.hidden_layer1 = np.tanh(np.dot(self.input, self.weights1))
        self.output_layer = np.tanh(np.dot(self.hidden_layer1, self.weights2))

        return self.output_layer

    def feed_backward(self):
        weights2_update = np.dot(self.hidden_layer1.T, 2 * (self.y - self.output_layer) * tanh_derivitave(self.output_layer))
        weights1_update = np.dot(self.input.T, np.dot(2 * (self.y - self.output_layer) * tanh_derivitave(self.output_layer),
                                                      self.weights2.T) * tanh_derivitave(self.hidden_layer1))
        self.weights1 += weights1_update
        self.weights2 += weights2_update

    def train(self):
        self.feed_forward()
        self.feed_backward()


nn = NeuralNetwork(X, y)

for i in range(25000):
    nn.train()

print("Actual Output: {}".format(y))
print("Predicted Output: {}".format(nn.feed_forward()))
print("Loss: {}".format(np.mean(np.square(y - nn.feed_forward()))))
