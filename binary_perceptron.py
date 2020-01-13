###############################################################################
# Madison Bowden
# MSU CSE891-003 Natural Language Processing SS2020
# HW1 - Multiclass Perceptron Text classifier
# Using sklearn fetch_20newsgroups_vectorized to load and transform data into
# tf-idf vectors
###############################################################################

import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
newsgroups_train = fetch_20newsgroups(subset='train')
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

class Perceptron(object):
    """
    Perceptron class that performs 1 vs All classification
    """
    def __init__(self, dim_input=30, dim_out=1, learning_rate=1):
        self.w = np.zeros((dim_input,dim_out))
        self.dim_input = dim_input
        self.dim_out = dim_out
        self.learning_rate = learning_rate

    def predict(self, input_array):
        y = input_array.dot(self.w)
        return np.sign(y)

    def one_update(self, input_array, label):
        y = self.predict(input_array) # get prediction for given input
        if y != label: # update weights only if prediction is wrong
            # self.w += input_array * (1 if label == 1 else -1)
            self.w += input_array

    def train(self, training_inputs, labels):
        """
        Go through all the training set and update weights
        param: training_inputs training set of input_arrays
        param: labels groundtruth for training set
        """
        for i in range(len(labels)):
            self.one_update(training_inputs[i], labels[i])

    def test(self, testing_inputs, labels):
        count_correct = 0
        pred_list = []
        for test_array, label in zip(testing_inputs,labels):
            prediction = perceptron.predict(test_array)
            if prediction == label:
                count_correct += 1
            pred_list.append(prediction)
        accuracy = float(count_correct)/len(labels)
        print('Accuracy is '+str(accuracy))
        return np.asarray(pred_list)


from sklearn.datasets import fetch_20newsgroups_vectorized
print("importing training and testing data")
training = fetch_20newsgroups_vectorized('train')
testing = fetch_20newsgroups_vectorized('test')
train_labels = [training.target[_]=='sci.space' for _ in range(len(training.target))]
test_labels = [testing.target[_]=='sci.space' for _ in range(len(testing.target))]
n_samples, n_features = training.data.shape
NUM_EPOCH = 50
print("done vectorizing data")
print()
print("beginning training")
perceptron = Perceptron(n_features)
t0 = time()
for ii in range(NUM_EPOCH):
    perceptron.train(training.data,train_labels)
t1 = time() - t0
print("training time:  %0.3fs" % t1)
print('For linear activation and '+str(NUM_EPOCH)+' epochs')

t0 = time()
pred_array = perceptron.test(testing.data,test_labels)
t1 = time() - t0
print("test time:  %0.3fs" % t1)
# print(metrics.confusion_matrix(test_labels, pred_array))
