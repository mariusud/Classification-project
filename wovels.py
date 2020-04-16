#!/usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal

def get_data():
    '''
    Returns data from the dataset
    '''
    data = np.genfromtxt('./wovels_dataset/vowdata_nohead.dat', dtype='U16',usecols=(7,8,9,10,11,12,13,14,15))
    classname = np.genfromtxt('./wovels_dataset/vowdata_nohead.dat', dtype='U16',usecols=(0))
    return data.astype(np.int), classname

def find_indeces(find, data):
    '''
    Finds the positions of a vowel in the data array
    '''
    return np.flatnonzero(np.core.defchararray.find(data,find)!=-1)

def sample_mean_vector(class_set):
    '''
    Returns the sample mean for a class set
    '''
    sample_mean = class_set.mean(axis=0)
    return sample_mean

def sample_covariance_matrix(sample_mean_vector, observations):
    '''
    Returns the sample covariance matrix for a class set
    '''
    dividend = 1/(len(observations)-1)
    difference = observations - sample_mean_vector
    tsum = np.dot(difference.T,difference)
    return tsum*dividend


class Gaussian_model():
    vowels= {'ae' : "had",
    'ah' : "hod",
    'aw' : "hawed",
    'eh' : "head",
    'er' : "heard",
    'ei' : "haid",
    'ih' : "hid",
    'iy' : "heed",
    'oa' : "/o/ as in boat",
    'oo' : "hood",
    'uh' : "hud", 
    'uw' : "who'd"
    }

    def __init__(self,data,classname):
        '''
        Initializes the class and splits data into train/test set
        '''
        self.x_train, self.y_train, self.x_test, self.y_test = self.split(data, classname)

    def split(self, data, classname):
        '''
        Splits the data set into train/test set.
        '''
        train, test = [], []
        for vowel in self.vowels:
            vowel_arr = find_indeces(vowel, classname)
            train.extend(vowel_arr[:70])
            test.extend(vowel_arr[70:])
        return data[train], classname[train], data[test], classname[test]

    def train(self):
        '''
        Model training.
        Model based on scipy's multivariate_normal
        '''
        probability_distribution = np.zeros((12,self.x_train.shape[0]))
        for i, vowel in enumerate(self.vowels):
            vowel_indeces = find_indeces(vowel,self.y_train) 
            vowel_samples = self.x_train[vowel_indeces]
            sample_mean = sample_mean_vector(vowel_samples)
            sample_covariance = sample_covariance_matrix(sample_mean,vowel_samples)

            self.rv = multivariate_normal(mean=sample_mean,cov=sample_covariance)
            probability_distribution[i] = self.rv.pdf(self.x_train)

        self.predicted_train = np.argmax(probability_distribution, axis=0)
        self.true_train = np.asarray([i for i in range(12) for _ in range(70)])

    def train_accuracy(self):
        print("Model accuracy for training set: ",np.sum(self.predicted_train==self.true_train)/len(self.predicted_train))

    def predict(self):
        '''
        Predicts vowel class based on the gaussian class model
        '''
        probability_distribution = np.zeros((12,self.x_test.shape[0]))
        for i, vowel in enumerate(self.vowels):
            vowel_indeces = find_indeces(vowel,self.y_test) 
            vowel_samples = self.x_test[vowel_indeces]
            sample_mean = sample_mean_vector(vowel_samples)
            sample_covariance = sample_covariance_matrix(sample_mean,vowel_samples)

            self.rv = multivariate_normal(mean=sample_mean,cov=sample_covariance)
            probability_distribution[i] = self.rv.pdf(self.x_test)

        self.predicted_test = np.argmax(probability_distribution, axis=0)
        self.true_test = np.asarray([i for i in range(12) for _ in range(69)])

    def prediction_accuracy(self):
        print("Model accuracy for testing set: ",np.sum(self.predicted_test==self.true_test)/len(self.predicted_test))

    def confusion_matrix(self, predictions, actual_class, features):
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

if __name__ == '__main__':
    data, classname = get_data()
    model = Gaussian_model(data,classname)
    model.train()
    model.predict()


    training_confusion = model.confusion_matrix(model.predicted_train,model.true_train,12)
    testing_confusion = model.confusion_matrix(model.predicted_test,model.true_test,12)
    print(training_confusion)
    print(testing_confusion)

    model.train_accuracy()
    model.prediction_accuracy()