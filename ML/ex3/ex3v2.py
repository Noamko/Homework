import sys
import numpy as np
import matplotlib.pyplot as plt


class NN:
    def __init__(self,inputSize, hiddenLayerCount, hiddenLayerSize, outputLayerSize):
        self.weights = []
        self.biases = []
        self.activations = []
        self.values = []
        # initialize weights and biases
        
        w = np.random.rand(inputSize, hiddenLayerSize)
        b = np.random.rand(hiddenLayerSize)
        self.biases.append(b)
        self.weights.append(w)

        for _ in range(hiddenLayerCount-1):
            w = np.random.rand(hiddenLayerSize, hiddenLayerSize)
            b = np.random.rand(hiddenLayerSize)
            self.weights.append(w)
            self.biases.append(b)

        w = np.random.rand(outputLayerSize, hiddenLayerSize)
        b = np.random.rand(outputLayerSize)

        self.weights.append(w)
        self.biases.append(b)
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def forward(self,x):
        activations = [x]
        zs = []
        activation = x
        for w,b in zip(self.weights[0], self.biases[0]):
            print(w)
            z = np.dot(w, activation) + b
        for i in range(len(self.weights[0])):
            z = np.dot(x, self.weights[0][i])
            a = self.sigmoid(z)
            self.activations.append(a)
            zs.append(z)
        self.activations.append(activation)
        self.values.append(zs)

        for i in range(1,len(self.weights)):
            for w in self.weights[i]:
                z = np.dot()
                a = self.sigmoid(z)
                activation.append(a)
                zs.append(z)
            self.activations.append(activation)
            self.values.append(zs)


train_x = np.loadtxt(sys.argv[1], max_rows=5000)
np.true_divide(train_x,255)
# train_y = np.loadtxt(sys.argv[2])
# test_x = np.loadtxt(sys.argv[3])

nn = NN(784,1,150,10)
nn.forward(train_x[0])

        