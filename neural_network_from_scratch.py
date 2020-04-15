import numpy as np

X = np.array(([0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]))
y = np.array(([1], [0], [0], [1]))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

class NeuralNetwork:
    def __init__(self, x, y):
        self.neurons_in_hidden_layer = 4
        self.input = x
        self.output = y
        self.weights1 = np.random.rand(X.shape[1], self.neurons_in_hidden_layer)
        self.weights2 = np.random.rand(self.neurons_in_hidden_layer, 1)

    def feed_forward(self):
        self.hidden_layer1 = tanh(np.dot(self.input, self.weights1))
        self.hidden_layer2 = tanh(np.dot(self.hidden_layer1, self.weights2))
        print(self.hidden_layer1)
        return self.hidden_layer2


nn = NeuralNetwork(X, y)
print(nn.feed_forward())

