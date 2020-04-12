#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

#The iris task

classmap = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

def arrange_order(x,t):
    '''
    Sets up the array such that the first 90 rows are 30*3 sets of different flowers
    '''
    array_x, array_t = [], []
    for i in range(50):
        for j in range(3):
            array_x.append(x[50*j+i])
            array_t.append(t[50*j+i])
    return np.array(array_x), np.array(array_t)

def get_x():
    f = open('./iris_dataset/iris.data')
    x = np.loadtxt(f,delimiter=',',usecols=(0,1,2,3))
    f.close()
    return x

def get_t():
    f = open('./iris_dataset/iris.data')
    t = np.loadtxt(f,delimiter=',', usecols=(4),dtype=str)
    f.close()
    t_updated = []
    for line in np.nditer(t):   
        t_updated.append(classmap.get(str(line)))
    return t_updated

def get_data(edit_order=True):
    x = get_x()
    t = get_t()
    if edit_order:
        x,t = arrange_order(x,t)
    return x,t

def MSE(A,B):
    '''
    Calculates the Mean-Squared Error between arrays A and B
    '''
    mse = (np.square(A - B)).mean(axis=1)
    return mse

def gradient_MSE(x, g, t):
    '''
    Calculates the gradient of the Mean-Squared Error
    '''
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
    W = np.zeros((3,4))
    mse_values = []
    for i in range(iterations):
        g = sigmoid(np.dot(x,W.T))
        W = W - alpha * gradient_MSE(x, g, t).T
        mse_values.append(MSE(g,t).mean())
    return W, mse_values 

def predict(W,x):
    g = np.dot(x, W.T)
    return np.argmax(sigmoid(g), axis=1)


def confusion_matrix(predictions, actual_class, features):
    '''
    Creates a confusion matrix for the classifier.
    predictions - predicted class.
    actual_class - what class the row is classes.
    features - the number of features predicted.
    '''
    confusion = np.zeros((features, features))
    for i in range(len(predictions)):
        confusion[actual_class[i]][predictions[i]] += 1
    return confusion.T

def print_histogram(data):
    sns.distplot(data[:,0],kde=True,norm_hist=True)
    sns.distplot(data[:,1],kde=True)
    sns.distplot(data[:,2],kde=True)
    sns.distplot(data[:,3],kde=True)
    title = "Histogram of four different flower features"
    plt.title(title)
    labels = 'Feature 1','Feature 2','Feature 3','Feature 4'
    plt.legend(labels)
    plt.savefig('histogram.png')
    plt.show()


if __name__ == '__main__':

    iterations = 100000
    alpha = 0.01
    x,t = get_data()

    W, mse_values = train(x[:90], t[:90], alpha, iterations)
    predictions = predict(W,x[90:])
    trueclass = np.argmax(t[90:], axis=1)
    print_histogram(x)
    confusion = confusion_matrix(predictions,trueclass,4)
    print("With 30 samples training and 20 for testing we get")
    print("correct guesses: ",100*np.sum(predictions == trueclass) / predictions.size, "%")
    print("Minimum mean-squared error obtained: ",mse_values[len(mse_values) -1])
    print("confusion matrix:\n",confusion)

    W, mse_values = train(x[60:], t[60:], alpha, iterations)
    predictions = predict(W,x[:60])
    trueclass = np.argmax(t[:60], axis=1)

    confusion = confusion_matrix(predictions,trueclass,4)
    print("With 20 samples training and 30 for testing we get")
    print("correct guesses: ",100*np.sum(predictions == trueclass) / predictions.size, "%")
    print("Minimum mean-squared error obtained: ",mse_values[len(mse_values) -1])
    print("confusion matrix:\n",confusion)


