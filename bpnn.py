import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
import scipy
from sklearn.model_selection import train_test_split
 
np.random.seed(1)

def sigmoid(Z):    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dA = np.squeeze(np.asarray(dA))
    s = np.squeeze(np.asarray(s))
    dZ = dA * s * (1-s)

    if (Z.shape[0] == 1):
        dZ = dZ.reshape((1, dZ.shape[0]))
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def relu(Z):    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_backward(dA, cache):    
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def initialize_parameters_deep(layer_dims):    
    np.random.seed(5)
    parameters = {}
    L = len(layer_dims)
 
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    # print(parameters)
    return parameters

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "sigmoid")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
 
    # Compute loss from aL and y.
    cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
 
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    for l in reversed(range(L-1)): 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "sigmoid")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
 
    return grads

def update_parameters(parameters, grads, learning_rate):    
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, filename='plot.png'):
    np.random.seed(1)
    costs = [] # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # print ('iteration:', i, '---------------------------')
        AL, caches = L_model_forward(X.T, parameters)
        # Compute cost.
        cost = compute_cost(AL, Y)
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (*100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.savefig(filename)

    return parameters

def getData(m):
    matrix = np.matrix(m).astype(int)
    labels = np.squeeze(np.asarray(matrix[:, -1]))
    dataset = matrix[:, 0:-1]
    return dataset, labels


def getData_test():
    # simulating coverage statement data
    dataset = np.matrix([
        [1,1,1,1,0,1,0,0,1],
        [1,0,0,0,1,1,1,1,0],
        [0,0,0,0,0,1,1,0,0],
        [1,1,0,0,1,0,1,1,1],
        [1,1,1,0,1,1,1,1,1],
        [0,0,1,0,0,1,1,1,0],
        [1,1,1,1,0,1,0,1,1]
    ]).astype(int)
    # simulating test cases results
    labels = np.array([0,0,0,0,0,1,1])
    # transform the labels to one-hot format
    labels_onehot = np.zeros((labels.shape[0], 2)).astype(int)
    labels_onehot[np.arange(len(labels)), labels.astype(int)] = 1
    # # divide the dataset into train and test datasets
    # train_dataset, test_dataset, \
    # train_labels, test_labels = train_test_split(
    #     dataset, labels, test_size = .1, random_state = 12)
    return dataset, labels

def getDataTest(dim):
    # estimate the suspiciousness of each statement
    test_susp_dataset = np.identity(dim)
    return test_susp_dataset

def forwardonly(X, parameters):    
    m = X.shape[1]
    n = len(parameters) // 2
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    return probas

def insertonSort(alist):
    index = [x for x in range(len(alist))]
    rank = [1 for x in range(len(alist))]
    for i in range(len(alist)):
        key = alist[i]
        val = index[i]
        j = i - 1 
        while j >= 0 and alist[j] > key:
            alist[j+1] = alist[j]
            index[j+1] = index[j]
            j -= 1
        alist[j+1] = key
        index[j+1] = val
    ranking = 1
    for i in range(len(alist)-1,0,-1):
        ranking += 1
        if alist[i] == alist[i-1]:
            rank[index[i-1]] = rank[index[i]]
        else:
            rank[index[i-1]] = ranking
    return rank,index

def train(train_dataset, train_labels):
    # set network structure
    # input layers: number of test cases
    # hidden layers: two hidden layers with 5 neurons each
    # output layers: one neuron
    layers_dims = [train_dataset.shape[1],5,5,1]
    train_labels = np.array([train_labels])
    parameters = L_layer_model(train_dataset, train_labels, layers_dims, learning_rate = 0.3, num_iterations = 7000, print_cost = True, filename='bpnn_learning_cost.png')
    return parameters

if __name__ == '__main__':
    train_dataset, train_labels = getData_test()
    
    params = train(train_dataset, train_labels)
    test_dataset = getDataTest(train_dataset.shape[1])
    result = forwardonly(test_dataset, params)
    print(result)
    rank, index= insertonSort(np.squeeze(np.asarray(result)))
    for i in range(len(rank)-1,-1,-1):
        print("Statement {:>2}: {:>4}".format(index[i]+1,rank[index[i]]))