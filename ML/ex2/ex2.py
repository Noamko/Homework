import numpy as np
from numpy.core.fromnumeric import transpose

def zscore(value, mean, s_dev):
    return (value - mean) / s_dev

def normalize_data(data):
    temp = []
    normalized = []
    for row in data:
        temp.append(np.array(row.split(','),float))
    for row in np.transpose(temp):
        r = []
        for value in row:
            r.append(zscore(value,np.mean(row),np.std(row)))
        normalized.append(r)
    return np.transpose(normalized) # TODO: check if we need to reutrn the transposed matrix or not
    
def KNN(value, data, classifications, k):
    # we assume that the classifications are indexed as same as the data_set
    neighbors = []
    i = 0
    for params in data:
        neighbors.append((np.linalg.norm(value - params)**2,classifications[i])) # should this be square?
        i += 1
    neighbors.sort(key=lambda x:x[0])
    neighbors = neighbors[::k]
    return np.bincount([i[1] for i in neighbors]).argmax()

dataset = np.loadtxt('train_x.txt',dtype='str')
normalized_data = normalize_data(dataset)
classes = np.loadtxt('train_y.txt',int)
test_dataset = normalize_data(np.loadtxt('test_x.txt',dtype='str'))
com = KNN(test_dataset[0],normalized_data,classes,19)
print(com)