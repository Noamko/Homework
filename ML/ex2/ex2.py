import numpy as np
import sys

def zscore(value, mean, s_dev):
    return (value - mean) / s_dev

def parse_data(arr):
    res = []
    for row in arr:
        res.append(np.array(row.split(','),float)) 
    return res

def normalize_data(data):
    normalized = []
    temp = parse_data(data)
    for row in np.transpose(temp):
        r = []
        for value in row:
            r.append(zscore(value,np.mean(row),np.std(row)))
        normalized.append(r)
    return np.transpose(normalized) # TODO: check if we need to reutrn the transposed matrix or not
    
def KNN(value, data, classifications, k):
    # we assume that the classifications are indexed as same as the data_set
    neighbors = []
    index = 0
    for params in data:
        neighbors.append((np.linalg.norm(value - params)**2, classifications[index])) # should this be square?
        index += 1
    neighbors.sort(key=lambda x:x[0]) # sort all neighbors by uclid distanc
    neighbors = neighbors[::k] # get the k first sorted neighbors (aka closests)
    return np.bincount([i[1] for i in neighbors]).argmax()

def multiclass_preceptron(value, training_x,training_y):
    w  = {}
    def argmax(x,w):
        temp = -1
        y_h = None
        for key in w:
            if temp < np.dot(x,w[key]):
                temp = np.dot(x,w[key])
                y_h = key
        return y_h

    for y in training_y: w[y] = np.zeros(len(value)) # init weight vectors for each class

    index = 0
    for x in training_x:
        y_hat = argmax(x,w)
        yi = training_y[index] # g(x) = yi --> true y 
        if y_hat != yi:
            w[yi] += x 
            w[y_hat] -= x
        index+=1
    return argmax(value, w)


training_x_set_path = sys.argv[1]
training_y_set_path = sys.argv[2]
test_x_set_path = sys.argv[3]
outfile_path = sys.argv[4]

training_set = np.loadtxt(training_x_set_path, dtype='str')
test_set = np.loadtxt(test_x_set_path, dtype='str')
classes = np.loadtxt(training_y_set_path, int)

normalized_training_data = normalize_data(training_set)
normalized_test_data = normalize_data(test_set)

for x in normalized_test_data:
    knn_yhat = KNN(x ,normalized_training_data,classes, 2)
    perceptron_yhat = multiclass_preceptron(x,normalized_training_data, classes)
    svm_yhat = -1
    pa_yhat = -1
    print(f"knn: {knn_yhat}, perceptron: {perceptron_yhat}, svm: {svm_yhat}, pa: {pa_yhat}\n")