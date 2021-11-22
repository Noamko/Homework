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
    def  train(self):
        learning_rate = 1
        epoch = 1000
        error = 0
        for y in self.training_y:
            self.w[y] = np.zeros(len(self.training_x[0])) # init weight vectors for each class
            self.biases[y] = 0
        for _ in range(epoch): # TODO: shuffle?
            error = 0
            for i,x in enumerate(self.training_x):
                x[4] = 0
                y_hat = argmax(x,self.w)
                yi = self.training_y[i] # g(x) = yi --> true y 
                if y_hat != yi:
                    self.w[yi] += x *learning_rate + self.biases[yi]
                    self.w[y_hat] -= x*learning_rate - self.biases[yi]
                    #self.biases[yi] += learning_rate 
                    error+=1
            #print((1 - error/ len(self.training_x)) * 100)

    def predict(self, v):

        return argmax(v, self.w)


class SVM:
    training_x = None
    training_y = None
    w = {}
    def __init__(self,training_x, training_y):
        self.training_x = training_x
        self.training_y = training_y
    
    def train(self):
        learning_rate = 1
        lambd = 0.01
        for y in self.training_y: self.w[y] = np.zeros(len(self.training_x[0])) # init weight vectors for each class
        index = 0
        for x in self.training_x:
            x[4] = 0
            y_hat = argmax(x,self.w)
            yi = self.training_y[index] # g(x) = yi --> true y
            if y_hat != yi:
                self.w[yi] = (1 - learning_rate * lambd) * self.w[yi] + learning_rate * x
                self.w[y_hat] = (1 - learning_rate * lambd) * self.w[y_hat] - learning_rate * x
            index+=1

    def predict(self,value):
        return argmax(value, self.w)
    

class PA:
    training_x = None
    training_y = None
    w = {}
    def __init__(self, training_x, training_y):
        self.training_x = training_x
        self.training_y = training_y

    def train(self):
        for y in self.training_y: self.w[y] = np.zeros(len(self.training_x[0])) # init weight vectors for each class
        index = 0
        for _ in range(20):
            index = 0
            for x in self.training_x:
                x[4] = 0
                y_hat = argmax(x,self.w)
                yi = self.training_y[index] # g(x) = yi --> true y
                if y_hat != yi:
                    tau = (max(0, (1 - np.dot(self.w[yi], x) + np.dot(self.w[y_hat],x)))) / (2 * np.linalg.norm(x)**2)
                    self.w[yi] = self.w[yi] + tau * x
                    self.w[y_hat] = self.w[y_hat] -  tau * x
                index+=1

    def predict(self,value):
        value[4] = 0
        return argmax(value, self.w)

classes = np.loadtxt(sys.argv[2], int)
normalized_training_data = normalize_data(np.loadtxt(sys.argv[1], dtype='str'))
out = open(sys.argv[4],'a')

knn = KNN(5,normalize_data(np.loadtxt(sys.argv[1], dtype='str'), 'zscore'), classes)
perceptron = Perceptron(normalize_data(np.loadtxt(sys.argv[1], dtype='str'), 'sigmoid'), classes)
pa = PA(normalize_data(np.loadtxt(sys.argv[1], dtype='str')), classes)
svm = SVM(normalize_data(np.loadtxt(sys.argv[1], dtype='str')), classes)

svm.train()
perceptron.train()
pa.train()

zscore_norm_test =  normalize_data(np.loadtxt(sys.argv[3], dtype='str'),'zscore')
sigmoid_norm_test =  normalize_data(np.loadtxt(sys.argv[3], dtype='str'),'sigmoid')
normalized_test = normalize_data(np.loadtxt(sys.argv[3], dtype='str'))

def print_test_results(Z,S,N):
    knn_res = []
    perceptron_res = []
    svm_res = []
    pa_res = []

    for x in Z:
        knn_res.append(knn.predict(x))
    for x in S:
        perceptron_res.append(perceptron.predict(x))
    for x in N:
        svm_res.append(svm.predict(x))
        pa_res.append(pa.predict(x))
    

    for i in range(len(N)):
        out.write(f"knn: {knn_res[i]}, perceptron: {perceptron_res[i]}, svm: {svm_res[i]}, pa: {pa_res[i]}\n")

print_test_results(zscore_norm_test,sigmoid_norm_test, normalized_test)



