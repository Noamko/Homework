# Noam Koren
# 308192871

import matplotlib.pyplot as plt
import numpy as np
import sys
# Ideas for higher performance:
# Add learning rate to perceptron 
# itarate more then once 

def zscore(value, mean, s_dev):
    return (value - mean) / s_dev

def parse_data(arr):
    res = []
    for row in arr:
        res.append(np.array(row.split(','),float)) 
    return res

def sigmoid(x):
    return  1 /(1 + np.exp(-x))

def normalize_data(data, fun = 'none'):
    normalized = []
    temp = parse_data(data)
    for row in np.transpose(temp):
        r = []
        for value in row:
            if fun == 'zscore': r.append(zscore(value,np.mean(row), np.std(row)))
            elif fun == 'sigmoid': r.append(sigmoid(value))
            else: r.append(value)
        normalized.append(r)
    return np.transpose(normalized)

def argmax(x,w):
    temp = -1
    y_h = None
    for key in w:
        if temp < np.dot(x,w[key]):
            temp = np.dot(x,w[key])
            y_h = key
    return y_h

class KNN:
    training_data = None
    classes = None;
    k = 2
    def __init__(self,k, training_data, classes):
        self.training_data = training_data
        self.classes = classes
        self.k = k

    def predict(self,value):
        value[4] = 0
        neighbors = []
        index = 0
        for params in self.training_data:
            params[4] = 0
            neighbors.append((np.linalg.norm(value - params), self.classes[index])) # should this be square?
            index += 1
        neighbors.sort(key=lambda x:x[0]) # sort all neighbors by uclid distanc
        neighbors = neighbors[0:self.k:] # get the k first sorted neighbors (aka closests)
        return np.bincount([i[1] for i in neighbors]).argmax()


class Perceptron:
    def __init__(self,X,Y):
        self.training_x = X
        self.training_y = Y
        self.w = {}
        self.biases = {}
    def fit(self):
        # lr of 0.001 and bias of 0.001 for all y gave 93.3%
        # TODO: check if diffrent biases for diffrenet labales increases
        learning_rate = 0.001
        epochs = 1000
        for y in self.training_y:
            self.w[y] = np.zeros(len(self.training_x[0])) # init weight vectors for each class
            self.biases[y] = -0.001
        for _ in range(epochs):
            for i,x in enumerate(self.training_x):
                x[4] = 0 # check if this matters
                y_hat = argmax(x,self.w)
                yi = self.training_y[i] # g(x) = yi --> true y 
                if y_hat != yi:
                    self.w[yi] = self.w[yi] + learning_rate * (x + self.biases[yi])
                    self.w[y_hat] = self.w[y_hat] - learning_rate * (x + self.biases[yi])

    def predict(self, v):
        return argmax(v, self.w)


class SVM:
    def __init__(self,X,Y):
            self.training_x = X
            self.training_y = Y
            self.w = {}
            self.biases = {}
    def fit(self):
        # lr of 0.001 and bias of 0.001 for all y gave 93.3%
        # TODO: check if diffrent biases for diffrenet labales increases
        learning_rate = 0.001
        lmbda = 0.0081
        epochs = 1000
        for y in self.training_y:
            self.w[y] = np.zeros(len(self.training_x[0])) # init weight vectors for each class
            self.biases[y] = -0.001
        for _ in range(epochs):
            for i,x in enumerate(self.training_x):
                x[4] = 0 # check if this matters
                y_hat = argmax(x,self.w)
                yi = self.training_y[i] # g(x) = yi --> true y 
                if y_hat != yi:
                    self.w[yi] = (1 - lmbda * learning_rate) * self.w[yi] + learning_rate * (x + self.biases[yi])
                    self.w[y_hat] = (1 - lmbda * learning_rate) * self.w[y_hat] - learning_rate * (x + self.biases[yi])

    def predict(self, v):
        return argmax(v, self.w)

class PA:
    def __init__(self,X,Y):
        self.training_x = X
        self.training_y = Y
        self.w = {}
        self.biases = {}
    def fit(self):
        # lr of 0.001 and bias of 0.001 for all y gave 93.3%
        # TODO: check if diffrent biases for diffrenet labales increases
        learning_rate = 0.01
        epochs = 1000
        for y in self.training_y:
            self.w[y] = np.zeros(len(self.training_x[0])) # init weight vectors for each class
            self.biases[0] = 0
            self.biases[1] = 0
            self.biases[2] = 0
        for _ in range(epochs):
            for i,x in enumerate(self.training_x):
                x[4] = 0 # noisy
                y_hat = argmax(x,self.w)
                yi = self.training_y[i] # g(x) = yi --> true y 
                if y_hat != yi:
                    tau = (max(0, (1 - np.dot(self.w[yi], x) + np.dot(self.w[y_hat],x)))) / (2*np.linalg.norm(x))
                    self.w[yi] = self.w[yi] + learning_rate * tau * (x + self.biases[yi])
                    self.w[y_hat] = self.w[y_hat] - learning_rate * tau * (x + self.biases[y_hat])

    def predict(self, v):
        v[4] = 0
        return argmax(v, self.w)

classes = np.loadtxt(sys.argv[2], int)
normalized_training_data = normalize_data(np.loadtxt(sys.argv[1], dtype='str'))
out = open(sys.argv[4],'a')

knn = KNN(5,normalize_data(np.loadtxt(sys.argv[1], dtype='str'), 'zscore'), classes)
perceptron = Perceptron(normalize_data(np.loadtxt(sys.argv[1], dtype='str'), 'sigmoid'), classes)
svm = SVM(normalize_data(np.loadtxt(sys.argv[1], dtype='str'),'sigmoid'), classes)
pa = PA(normalize_data(np.loadtxt(sys.argv[1], dtype='str'),'zscore'), classes)

svm.fit()
perceptron.fit()
pa.fit()

zscore_norm_test =  normalize_data(np.loadtxt(sys.argv[3], dtype='str'),'zscore')
sigmoid_norm_test =  normalize_data(np.loadtxt(sys.argv[3], dtype='str'),'sigmoid')
normalized_test = normalize_data(np.loadtxt(sys.argv[3], dtype='str'))

def print_test_results(Z,S,N):
    knn_res = []
    perceptron_res = []
    svm_res = []
    pa_res = []

    for x in Z: # zscore normalization
        knn_res.append(knn.predict(x))
        pa_res.append(pa.predict(x))
    for x in S: # sigmoid normalization
        perceptron_res.append(perceptron.predict(x))
        svm_res.append(svm.predict(x))
    for x in N: # no normalization
        pass
    for i in range(len(N)):
        out.write(f"knn: {knn_res[i]}, perceptron: {perceptron_res[i]}, svm: {svm_res[i]}, pa: {pa_res[i]}\n")

print_test_results(zscore_norm_test,sigmoid_norm_test, normalized_test)



