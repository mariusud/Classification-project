#!/usr/bin/env python3

import numpy as np
import matplotlib as plt

#The iris task

classmap = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

def get_data(cutting_position):
    '''cutting_position: where to cut the training and test set e.g. 30/20'''

    x, t = [], []
    f = open('./iris_dataset/iris.data')
    line = f.readline()
    while line:
        line = line.strip('\n')
        if line == '':  
            continue
        line = line.split(',')
        x.append(np.array([float(i) for i in line[:-1]]))
        t.append(classmap.get(line[-1]))
        line = f.readline()
    f.close()

    array_x, array_t = [], []

    for i in range(50):
        for j in range(3):
            array_x.append(x[50*j+i])
            array_t.append(t[50*j+i])

    return np.array(array_x), np.array(array_t)


def MSE(A,B):
    '''
    Calculates the Mean-Squared Error between arrays A and B
    '''
    mse = (np.square(A - B)).mean(axis=1)
    return mse

def gradient_MSE(x, g, t):
    mse_gradient = g - t
    g_gradient = g * (1-g)
    zk_gradient = x.T
    return np.dot(zk_gradient, mse_gradient*g_gradient)

def sigmoid(x):
    '''returns the sigmoid, an acceptable approximation of the heaviside function'''
    return (1/(1+np.exp(-x)))

    
def train(x, t, alpha, iterations):
    '''
    x - training data,
    t - true class,
    alpha - step factor,
    W - weight matrix,
    g - output vector,
    '''
    W = np.random.normal(0, 1, (3, x.shape[1]))
    mse_values = []
    for i in range(iterations):
        g = sigmoid(np.dot(x,W.T))
        W = W - alpha * gradient_MSE(x, g, t).T
        mse_values.append(MSE(g,t))
    return W, mse_values 


def predict(W,x):
    g = np.dot(x, W.T)
    return g




if __name__ == '__main__':

    iterations = 100
    alpha = 0.2

    x,t = get_data(30)
    W, mse_values = train(x[:90], t[:90], alpha, iterations)

    predictions = predict(W,x[90:])
    print(predictions)

