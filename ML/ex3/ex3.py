# Noam Koren
# 308192871
import sys
import numpy as np
import matplotlib.pyplot as plt

def digit_to_array(d):
    arr = np.zeros(10)
    arr[int(d)] = 1
    return arr
def shuffle(x, y):
    np.random.get_state()
    rand_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rand_state)
    np.random.shuffle(y)
    return x, y

class NN:
    #initialize weights and biases
    def __init__(self,hiddenLayerSize):
        self.w1 = np.random.uniform(-0.09, 0.09, [hiddenLayerSize, 784])
        self.b1 = np.random.uniform(-0.09, 0.09, [hiddenLayerSize, 1])  # 0.25 -> 15   0.9-0.9->13 0.09-9.6
        self.w2 = np.random.uniform(-0.09, 0.09, [10, hiddenLayerSize])
        self.b2 = np.random.uniform(-0.09, 0.09, [10, 1])
        self.epochs = 20

    def forward(self,x):
        x = x.reshape(len(x), 1)
        z1 = np.dot(self.w1, x) + self.b1
        h1 = self.sigmoid(z1)
        z2 = np.dot(self.w2, h1) + self.b2
        a = self.softmax(z2)
        return (x,z1,h1,z2,a)

    def backprob(self,f,y):
        x, z1, h1, z2, a = f
        dz2 = self.softmax(z2).reshape(10,1)
        dz2[int(y)] -= 1
        dw2 = np.matmul(dz2, h1.T)
        db2 = dz2
        dz1 = np.matmul(self.w2.T, dz2) * self.sigmoid_derivative(z1)
        dw1 = np.matmul(dz1, x.T)
        db1 = dz1
        return (dw1, db1, dw2, db2)

    def softmax(self,x):
        x = x - np.max(x)
        return np.exp(x) / sum(np.exp(x))
    def sigmoid(self, v):
        v = np.clip(v,-500, 500)
        return np.array([1 / (1 + np.exp(-x)) for x in v])
    
    def sigmoid_derivative(self, x):
        a = self.sigmoid(x)
        return a * (1 - a)
    

    def train(self,x,y):
        training_set = list(zip(x,y))
        for i in range(self.epochs):
            np.random.shuffle(training_set)
            print("training: epoch: " + str(i) + "/20")
            for example, label in training_set:
                dw1, db1, dw2 ,db2 = self.backprob(self.forward(example),label)
                self.w1 = self.w1 - 0.02 * dw1
                self.b1 = self.b1 - 0.02 * db1

                self.w2 = self.w2 - 0.02 * dw2
                self.b2 = self.b2 - 0.02 * db2
    def predict(self,x):
        res = self.forward(x)[4]
        max = 0
        result = 0
        for i in range(len(res)):
            if max < res[i]:
                max = i
                result = i
        return result

            
print("loading training data...")
train_x = np.loadtxt(sys.argv[1])
train_x = np.true_divide(train_x, 255)
train_y = np.loadtxt(sys.argv[2])
test_x = np.loadtxt(sys.argv[3])
print("done loading training data.")

print("shuffeling & cutting.")
training_set = list(zip(train_x,train_y))
np.random.shuffle(training_set)
training_set = training_set[:5000]
print("done.")

nn = NN(150)
x,y = training_set
nn.train(x, y)
file = open("test_y", 'w')
for i in range(len(test_x)):
    print("Testing:" + str(i))
    x, z1, h1, z2, a  = nn.forward(test_x[i])
    prediction = a.argmax()
    file.write(f"{prediction}\n")
file.close()