#############################################################################
# Madison Bowden | NLP SS2020
# MultiClass Perceptron Classifier
#############################################################################

import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
newsgroups_train = fetch_20newsgroups(subset='train')
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
BIAS = 1
NUM_EPOCH = 50

class Perceptron(object):
    """
    MultiClass Perceptron classifier that performs 1 vs All classification
    with linear winner takes all activation
    """
    def __init__(self, dim_input=30, dim_out=1, learning_rate=.5):
        self.w = np.zeros((dim_out,dim_input,1)) # weight vector for n classes
        self.bias = np.zeros(dim_out) # Biases added, but performed worse, so removed from predict & train
        self.dim_input = dim_input # number of features
        self.dim_out = dim_out # n classes
        self.learning_rate = learning_rate

    def predict(self, input_array):
        """
        returns max categorization from list of w.x + b for each classifier
        param: input_array of size 1 x dim_input
        """
        y = [self.bias[_] + input_array.dot(self.w[_]) for _ in range(self.dim_out)]
        return np.argmax(y)

    def train(self, training_inputs, labels):
        """
        Go through all the training set and update weights
        param: training_inputs training set of input_arrays
        param: labels groundtruth for training set
        """
        for i in range(len(labels)):
            y = self.predict(training_inputs[i]) # get prediction for given input
            if y != labels[i]: # update weights only if prediction is wrong
                self.w[y] = self.w[y] - self.learning_rate*training_inputs[i].transpose()
                # self.w[y] -= self.learning_rate*BIAS
                self.w[labels[i]] = self.w[labels[i]] + self.learning_rate*training_inputs[i].transpose()
                # self.w[labels[i]] += self.learning_rate*BIAS

    def test(self, testing_inputs, labels):
        """
        prints top 1 accuracy and returns prediction list
        """
        count_correct = 0
        pred_list = []
        for test_array, label in zip(testing_inputs,labels):
            prediction = perceptron.predict(test_array)
            if prediction == label:
                count_correct += 1
            pred_list.append(prediction)
        accuracy = float(count_correct/len(labels))
        print('Accuracy is '+str(accuracy))
        return np.asarray(pred_list)

# MAIN
from sklearn.datasets import fetch_20newsgroups_vectorized
# set.data set.target (numbers) set.target_names
print("importing training and testing data")
training = fetch_20newsgroups_vectorized('train')
testing = fetch_20newsgroups_vectorized('test')
train_labels = training.target
test_labels = testing.target
n_samples, n_features = training.data.shape

print("done vectorizing data")
print()
print("beginning training")
perceptron = Perceptron(n_features,len(training.target_names))
t0 = time()
for ii in range(NUM_EPOCH):
    perceptron.train(training.data,train_labels)
    print('epoch: ',ii)
t1 = time() - t0
print("training time:  %0.3fs" % t1)
print('For linear activation and '+str(NUM_EPOCH)+' epochs')

t0 = time()
pred_array = perceptron.test(testing.data,test_labels)
t1 = time() - t0
print("test time:  %0.3fs" % t1)
