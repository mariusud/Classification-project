#!/usr/bin/env python3
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


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

    def __init__(self,data,classname,diagonal=False):
        '''
        Initializes the class and splits data into train/test set
        '''
        self.x_train, self.y_train, self.x_test, self.y_test = self.split(data, classname)
        self.diagonal = diagonal

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
            if self.diagonal:
                sample_covariance = np.diag(np.diag(sample_covariance))
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
            if self.diagonal:
                sample_covariance = np.diag(np.diag(sample_covariance))
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


class GaussianMixture_model():
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

    def __init__(self, data, classname,components, diagonal=False):
        '''
        Initializes the class and splits data into train/test set
        '''
        self.x_train, self.y_train, self.x_test, self.y_test = self.split(data, classname)
        self.diagonal = diagonal
        self.components = components
        self.GMMs =  dict.fromkeys(range(len(self.vowels)))


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
        Model based on sklearn's Gaussian Mixture
        '''
        probability_distribution = np.zeros((12,self.x_train.shape[0]))
        for i, vowel in enumerate(self.vowels):
            vowel_indeces = find_indeces(vowel,self.y_train) 
            vowel_samples = self.x_train[vowel_indeces]
            sample_mean = sample_mean_vector(vowel_samples)
            sample_covariance = sample_covariance_matrix(sample_mean,vowel_samples)
            if self.diagonal:
                sample_covariance = np.diag(np.diag(sample_covariance))
            self.GMMs[i] = GaussianMixture(n_components=self.components, covariance_type='diag',max_iter=500000)
            self.GMMs[i].fit(vowel_samples, self.y_train[vowel_indeces])
            probability_distribution[i] = self.GMMs[i].score_samples(self.x_train)

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
            probability_distribution[i] = self.GMMs[i].score_samples(self.x_test)
        
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

def problem_1(data,classname):
    model = Gaussian_model(data,classname)
    model.train()
    model.predict()


    training_confusion = model.confusion_matrix(model.predicted_train,model.true_train,12)
    testing_confusion = model.confusion_matrix(model.predicted_test,model.true_test,12)
    print(training_confusion)
    print(testing_confusion)

    model.train_accuracy()
    model.prediction_accuracy()

    print("\ndiagonal model\n")
    diagonal_model = Gaussian_model(data,classname,True)
    diagonal_model.train()
    diagonal_model.predict()

    diag_training_confusion = diagonal_model.confusion_matrix(diagonal_model.predicted_train,diagonal_model.true_train,12)
    diag_testing_confusion = diagonal_model.confusion_matrix(diagonal_model.predicted_test,diagonal_model.true_test,12)
    print(diag_training_confusion)
    print(diag_testing_confusion)
    diagonal_model.train_accuracy()
    diagonal_model.prediction_accuracy()

def problem_2(data,classname):
    gmm3 = GaussianMixture_model(data,classname,3)
    gmm2 = GaussianMixture_model(data,classname,2)

    gmm2.train()
    gmm2.predict()
    gmm2.train_accuracy()
    gmm2.prediction_accuracy()
    training_confusion_gmm2 = gmm2.confusion_matrix(gmm2.predicted_train,gmm2.true_train,12)
    print(training_confusion_gmm2)

    gmm3.train()
    gmm3.predict()
    gmm3.train_accuracy()
    gmm3.prediction_accuracy()
    training_confusion_gmm3 = gmm3.confusion_matrix(gmm3.predicted_train,gmm3.true_train,12)
    print(training_confusion_gmm3)


if __name__ == '__main__':
    data, classname = get_data()
    problem_1(data,classname)
    #problem_2(data,classname)
