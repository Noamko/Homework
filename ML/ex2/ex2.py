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

def normalize_data(data):
    normalized = []
    temp = parse_data(data)
    for row in np.transpose(temp):
        r = []
        for value in row:
            # r.append(sigmoid(value))
            r.append(zscore(value,np.mean(row), np.std(row)))
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


class Preceptron:
    training_x = None
    training_y = None
    w = {}
    def __init__(self,training_x,training_y):
        self.training_x = training_x
        self.training_y = training_y
        
    def train(self):
        learning_rate = 1
        iterations = 1
        for y in self.training_y: self.w[y] = np.zeros(len(self.training_x[0])) # init weight vectors for each class
        for _ in range(iterations): # TODO: shuffle?
            index = 0
            for x in self.training_x:
                x[4] = 0
                y_hat = argmax(x,self.w)
                yi = self.training_y[index] # g(x) = yi --> true y 
                if y_hat != yi:
                    self.w[yi] += learning_rate * x
                    self.w[y_hat] -= learning_rate * x
                index+=1
    def predict(self, v):
        v[4] = 0
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


training_x_set_path = sys.argv[1]
training_y_set_path = sys.argv[2]
test_x_set_path = sys.argv[3]
outfile_path = sys.argv[4]

training_set = np.loadtxt(training_x_set_path, dtype='str')
test_set = np.loadtxt(test_x_set_path, dtype='str')
classes = np.loadtxt(training_y_set_path, int)

normalized_training_data = normalize_data(training_set)
normalized_test_data = normalize_data(test_set)

out = open(outfile_path,'a')

knn = KNN(5, normalized_training_data, classes)
preceptron = Preceptron(normalized_training_data, classes)
pa = PA(normalized_training_data, classes)
svm = SVM(normalized_training_data, classes)

svm.train()
preceptron.train()
pa.train()

for x in normalized_test_data:
    knn_yhat = knn.predict(x)
    perceptron_yhat = preceptron.predict(x)
    svm_yhat =  svm.predict(x)
    pa_yhat = pa.predict(x)
    out.write(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")




