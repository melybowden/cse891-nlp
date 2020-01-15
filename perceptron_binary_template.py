#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random


# Get the sklearn digit recognition toy dataset, it contains 569 cases.
# We then take the first 500 to be our training set and the last 69 for testing

# In[2]:


# load the data set
img,label=sklearn.datasets.load_breast_cancer(return_X_y=True)
# split the data set
TRAIN_SIZE = 500
label = 2*label-1
train_img,test_img = img[:TRAIN_SIZE], img[TRAIN_SIZE:]
train_label,test_label = label[:TRAIN_SIZE], label[TRAIN_SIZE:]


# In[3]:


# Perceptron Class
class Perceptron(object):
    # Initialize the perceptron. Default values are set to 30, 1, & 1 for cancer dataset or inputs for digit classifier
    def __init__(self, dim_input=30, dim_out=1,learning_rate=1):
        self.w = np.zeros((dim_out, dim_input)) # array of weights for all output dimensions
        self.dim_input = dim_input # dimension of the input (30 for our medical cases)
        self.learning_rate = learning_rate

    def predict(self,input_array):
        """
        Predicts a label for a given input array
        param: input_array of size dim_input
        return: a single label output
        """
        arg_max = 0
        y = [self.w[_].dot(input_array) for _ in range(self.w.shape[0])]

        # y = []
        # for i in range(self.w.shape[0]):
        #     y.append(self.w[i].dot(input_array))
        # Equivalent to above lines
        # y = [self.w[_,:].dot(input_array) for _ in range(self.w.shape[0])] # z =  W * x
        if self.w.shape[0] == 1:
            return np.sign(y) # return sign{z}
        else: # For n-ary return max classification
            y = np.arange(self.w.shape[0])[y == np.max(y)]
            if y.size > 1: # Return only 1 label prediction
                y = y[0]
            return y


    def one_update(self,input_array,label):
        """
        Updates weights (w) for given input_array based on the prediction and groundtruth
        param: input_array of size dim_input
        param: label groundtruth
        """
        y = self.predict(input_array) # get prediction for given input
        if y != label: # update weights only if prediction is wrong (not groundtruth)
            if self.w.shape[0] == 1: # update + or - x for binary
                self.w += input_array * (1 if label == 1 else -1)
            else: # update weights for n-ary classification
                self.w[label] += input_array
                self.w[y] -= input_array

    def train(self, training_inputs, labels):
        """
        Go through all the training set and update weights
        param: training_inputs training set of input_arrays
        param: labels groundtruth for training set
        """
        for i in range(len(labels)):
            self.one_update(training_inputs[i], labels[i])

    def test(self, testing_inputs, labels):
        # number of correct predictions
        count_correct = 0
        # a list of the predicted labels the same order as the input
        pred_list = []
        for test_array, label in zip(testing_inputs,labels):
            prediction = perceptron.predict(test_array)
            if prediction == label:
                count_correct += 1
            pred_list.append(prediction)
        accuracy = float(count_correct)/len(test_label)
        print('Accuracy is '+str(accuracy))
        return np.asarray(pred_list)


# In[4]:


# Number of epochs (iterations over the training set)
NUM_EPOCH = 6


# In[5]:


# Try the perceptron with sigmoid activation
perceptron = Perceptron()
for ii in range(NUM_EPOCH):
    perceptron.train(train_img, train_label)
print('For sigmoid activation and '+str(NUM_EPOCH)+' epochs')
pred_array = perceptron.test(test_img, test_label)


# In[6]:


# Confusion matrix shows what we predicted vs what was the real (True) label.
# A perfect classifier will have has non zero elements only in the diagonal (why??)
# Look at the results outside the diagonal, does it make sense that these mistakes happened?
confusion_mat = confusion_matrix((test_label+1)/2, (pred_array+1)/2, labels=range(0,2))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cax = ax.matshow(confusion_mat)
plt.title('Confusion Matrix\n')
fig.colorbar(cax)
labels = ['malignant', 'benign']
ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[7]:


confusion_mat


# # Extra Credit!!!
# The same simple rule can be applied to multiple classes, update your code to work with any number of classes!

# In[11]:


# load the data set
# This is handwritten digit recognition
img,label=sklearn.datasets.load_digits(return_X_y=True)
TRAIN_SIZE = 1700
# split the data set
train_img,test_img = img[:TRAIN_SIZE], img[TRAIN_SIZE:]
train_label,test_label = label[:TRAIN_SIZE], label[TRAIN_SIZE:]


# As can be observed, each of these train img is an 8x8 pixel grayscale image of a handwritten digit,
# for instance training image number 47 is of the handwritten digit '1'.
# We can also verify that the label in the dataset is indeed 1.

# In[12]:


IMG_DIM = (8,8)


# In[13]:


# Try it yourself with any index!
img_idx = 2
plt.matshow(np.reshape(train_img[img_idx],IMG_DIM),cmap='gray')
print('label in the dataset is '+str(train_label[img_idx]))


# In[19]:


perceptron = Perceptron(dim_input=8*8, dim_out=10)
for ii in range(NUM_EPOCH):
    perceptron.train(train_img,train_label)
print('For linear activation and '+str(NUM_EPOCH)+' epochs')
pred_array = perceptron.test(test_img,test_label)


# In[20]:


#########################################################################################
# Confusion matrix shows what we predicted vs what was the real (True) label.
# A perfect classifier will have has non zero elements only in the diagonal (why??)
# Look at the results outside the diagonal, does it make sense that these mistakes happened?
confusion_mat = confusion_matrix(test_label, pred_array, labels=range(0,10))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cax = ax.matshow(confusion_mat)
plt.title('Confusion Matrix\n')
fig.colorbar(cax)
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ax.set_xticklabels(['']+labels)
ax.set_yticklabels(['']+labels)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[21]:


# Note the perceptron seems to have misclassified some 3s as 8s, these digits do look similar
# so this is to be expected.. what else did we misclassify?

# 9's and 5's as well as 4's and 7's were sometimes misclassified, since those sometimes look similar

# In[ ]:





# In[ ]:
