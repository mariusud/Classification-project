#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
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
    #[x^T 1]^T -> x
    x_new = np.ones([np.size(x,0),np.size(x,1)+1])
    x_new[:,:-1] = x
    return x_new

def get_t():
    f = open('./iris_dataset/iris.data')
    t = np.loadtxt(f,delimiter=',', usecols=(4),dtype=str)
    f.close()
    t_updated = []
    for line in np.nditer(t):   
        t_updated.append(classmap.get(str(line)))
    return t_updated

def get_data(edit_order=True):
    '''
    Returns data from the dataset in mixed or correct order
    '''
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
    W = np.zeros((3,x.shape[1]))
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
    return confusion

def print_histogram():
    data = get_x()
    for i in range(0,4):
        plt.figure(i)
        for j in range(0,3):
            sns.distplot(data[50*j:50*(j+1),i],kde=True,norm_hist=True)
        labels = 'Class 1','Class 2','Class 3'
        title = "Histogram of different classes for feature " +  str(i+1)
        plt.legend(labels)
        plt.title(title) 
        plt.savefig('./figures/histogram' + str(i+1) + '.png')
    plt.show()

def plot_mse(iterations, mse_values, mse_values2, mse_values3):
    plt.figure(4)
    plt.title("Fluctuations and development of MSE values with respect to alpha")
    steps = list(range(iterations))
    plt.plot(steps, mse_values3, steps, mse_values2, steps, mse_values)
    labels = "alpha = 0.05", "alpha = 0.02", "alpha = 0.005"
    plt.legend(labels)
    plt.ylim(0, 0.5)
    plt.ylabel('mean-square error')
    plt.xlabel('iteration')
    plt.savefig("./figures/mse_values.png")
    plt.show()

def print_error_rate(predictions, trueclass):
    print("Error rate: ", 100*(1 - np.sum(predictions == trueclass) / predictions.size), "%")

def problem1(x,t,alpha, iterations):
    W, mse_values = train(x[:90], t[:90], alpha, iterations)
    print("Problem 1\n W= \n", W)
    predictions = predict(W,x[90:])
    train_values = predict(W,x[:90])
    trueclass = np.argmax(t[90:], axis=1)
    trueclass_train = np.argmax(t[:90], axis=1)

    confusion_test = confusion_matrix(predictions,trueclass,3)
    confusion_train = confusion_matrix(train_values,trueclass_train,3)

    print("With 30 samples training and 20 for testing we get")
    print("Minimum mean-squared error obtained: ",mse_values[len(mse_values) -1])
    print("For the test set: ")
    print_error_rate(predictions, trueclass)
    print("confusion matrix:\n",confusion_test)
    print("For the training set: ")
    print_error_rate(train_values, trueclass_train)
    print("confusion matrix:\n",confusion_train)

    W, mse_values = train(x[60:], t[60:], alpha, iterations)
    predictions = predict(W,x[:60])
    train_values = predict(W,x[:90])
    trueclass = np.argmax(t[:60], axis=1)

    confusion_test = confusion_matrix(predictions,trueclass,3)
    confusion_train = confusion_matrix(train_values,trueclass_train,3)

    print("With last 30 samples training and first 20 for testing we get")
    print("Minimum mean-squared error obtained: ",mse_values[len(mse_values) -1])
    print("For the test set: ")
    print_error_rate(predictions, trueclass)
    print("confusion matrix:\n",confusion_test)
    print("For the training set: ")
    print_error_rate(train_values, trueclass_train)
    print("confusion matrix:\n",confusion_train)

    #Testing different alpha's to plot MSE
    W2, mse_values2 = train(x[:90], t[:90], 0.02, iterations)
    W3, mse_values3 = train(x[:90], t[:90], 0.05, iterations)
    plot_mse(iterations,mse_values, mse_values2, mse_values3)

def problem2(x,t,alpha,iterations):
    #Remove feature 2
    x = np.delete(x, 1, axis=1)    
    W, mse_values = train(x[:90], t[:90], alpha, iterations)
    predictions = predict(W,x[90:])
    trueclass = np.argmax(t[90:], axis=1)
    train_values = predict(W,x[:90])
    trueclass_train = np.argmax(t[:90], axis=1)
    confusion_test = confusion_matrix(predictions,trueclass,3)
    confusion_train = confusion_matrix(train_values,trueclass_train,3)

    print("\nProblem 2\n\nRemoved 1 feature")
    print_error_rate(predictions, trueclass)
    print("Minimum mean-squared error obtained: ",mse_values[len(mse_values) -1])
    print("confusion matrix for test set:\n",confusion_test)

    #Remove another feature
    x = np.delete(x, 2, axis=1)    
    W, mse_values = train(x[:90], t[:90], alpha, iterations)
    predictions = predict(W,x[90:])
    trueclass = np.argmax(t[90:], axis=1)
    confusion_test = confusion_matrix(predictions,trueclass,3)
    train_values = predict(W,x[:90])
    trueclass_train = np.argmax(t[:90], axis=1)
    confusion_train = confusion_matrix(train_values,trueclass_train,3)

    print("\nRemoved 2 features")
    print_error_rate(predictions, trueclass)
    print("Minimum mean-squared error obtained: ",mse_values[len(mse_values) -1])
    print("confusion matrix for test set:\n",confusion_test)

    #Remove third feature
    x = np.delete(x, 0, axis=1)   
    W, mse_values = train(x[:90], t[:90], alpha, iterations)
    predictions = predict(W,x[90:])
    trueclass = np.argmax(t[90:], axis=1)
    confusion_test = confusion_matrix(predictions,trueclass,3)
    train_values = predict(W,x[:90])
    trueclass_train = np.argmax(t[:90], axis=1)
    confusion_train = confusion_matrix(train_values,trueclass_train,3)

    print("\nRemoved 3 features")
    print_error_rate(predictions, trueclass)
    print("Minimum mean-squared error obtained: ",mse_values[len(mse_values) -1])
    print("confusion matrix for test set:\n",confusion_test)
    print("confusion matrix for training set:\n",confusion_train)

if __name__ == '__main__':
    iterations = 2500
    alpha = 0.005
    x,t = get_data()
    
    print_histogram()
    problem1(x,t,alpha,iterations)
    problem2(x,t,alpha,iterations)
