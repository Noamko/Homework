# Noam Koren
# 308192871
import sys
import numpy as np

"""TODO:
refactor to be more generic
"""


class NN:
    # initialize weights and biases
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize, hiddenLayerCount = 1):
        self.epochs = 24
        self.learning_rate = 0.019
        self.weights = [np.random.uniform(-0.09, 0.09, [hiddenLayerSize, inputLayerSize])]
        self.biases = [np.random.uniform(-0.09, 0.09, [hiddenLayerSize, 1])]
        for i in range(0, hiddenLayerCount):
            self.weights.append(np.random.uniform(-0.09, 0.09, [hiddenLayerSize, hiddenLayerSize]))
            self.biases.append(np.random.uniform(-0.09, 0.09, [hiddenLayerSize, 1]))

        self.weights[-1] = np.random.uniform(-0.09, 0.09, [outputLayerSize, hiddenLayerSize])
        self.biases[-1] = np.random.uniform(-0.09, 0.09, [outputLayerSize, 1])

    def set_epoch_count(self, c):
        self.epochs = c
    def set_learning_rate(self, x):
        self.learning_rate = x

    def forward(self, x):
        x = x.reshape(len(x), 1)
        zs = [x]
        hs = [x]
        z1 = np.dot(self.weights[0], x) + self.biases[0]
        zs.append(z1)
        hs.append(self.sigmoid(z1))

        for w, b in zip(self.weights[1:-1], self.biases[1:-1]):
            z = np.dot(w, hs[-1]) + b
            zs.append(z)
            hs.append(self.sigmoid(z))
        z2 = np.dot(self.weights[-1], hs[-1]) + self.biases[-1]
        zs.append(z2)
        a = self.softmax(z2)
        hs.append(a)
        return (zs, hs)

    def backprob(self, forward_result, y):
        dws = [np.zeros(len(self.weights)) for w in self.weights]
        dbs = [np.zeros(len(self.weights)) for b in self.biases]
        zs, hs = forward_result
        _y = np.zeros(10)
        _y[int(y)] = 1
        dz2 = 2*(hs[-1] - _y.reshape(10, 1))  # TODO: check what is the real derivative for now this works fine
        dws[-1] = np.matmul(dz2, hs[-2].T)
        dbs[-1] = dz2
        dz = dz2
        for i in range(2, len(self.weights)+1):
            dz = np.matmul(self.weights[-i+1].T, dz) * self.sigmoid_derivative(zs[-i])
            dws[-i] = np.matmul(dz, hs[-i-1].T)
            dbs[-i] = dz
        return (dws, dbs)

    def softmax(self, x):
        x = x - np.max(x)
        e_x = np.exp(x)
        return e_x / sum(np.exp(x))

    def sigmoid(self, v):
        v = np.clip(v, -500, 500)
        return np.array([1 / (1 + np.exp(-x)) for x in v])

    def sigmoid_derivative(self, x):
        a = self.sigmoid(x)
        return a * (1 - a)

    def train(self, x, y):
        training_set = list(zip(x, y))
        for i in range(self.epochs):
            np.random.shuffle(training_set)
            print("training: epoch: " + str(i) + "/20")
            for example, label in training_set:
                dws, dbs = self.backprob(self.forward(example), label)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * dws[i]
                    self.biases[i] -= self.learning_rate * dbs[i]
        print("done.")
    def predict(self, x):
        prediction_vector = nn.forward(x)[1][-1]
        return prediction_vector.argmax()


print("loading training data...")
train_x = np.loadtxt(sys.argv[1])
train_x = np.true_divide(train_x, 255)
train_y = np.loadtxt(sys.argv[2])
test_x = np.loadtxt(sys.argv[3])
test_x = np.true_divide(test_x, 255)
print("done loading training data.")

print("shuffeling & cutting.")
training_set = list(zip(train_x, train_y))
np.random.shuffle(training_set)
training_set = training_set[:5000]
print("done.")

nn = NN(784, 150, 10, 3)
x, y = zip(*training_set)
nn.train(x, y)
file = open("test_y", 'w')
for i in range(len(test_x)):
    print("Testing:" + str(i))
    prediction = nn.predict(test_x[i])
    file.write(f"{prediction}\n")
file.close()