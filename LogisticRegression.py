from __future__ import division
import numpy as np
from numpy import matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
import math
from sklearn.preprocessing import normalize

np.random.seed(1000)

class LogisticRegression:

    def __init__(self, datasets):
        # a python list containing the names of datasets run logistic regression on
        self.datasets = datasets

        # use to store weights found after optimization
        self.w = {}
        self.w['analytic_grad'] = {}
        self.w['keras'] = {}

        self.elementwise_sig = np.vectorize(self.sigmoid)
        self.elementwise_get_class = np.vectorize(self.get_class)

    def run(self):
        """Do logistic regression 2 ways"""
        for dataset in self.datasets:
            self.get_dataset(dataset)
            self.set_parameters(dataset)

            # with gradient descent using analytically computed gradients
            print(dataset + ", ANALYTIC GRADIENT")
            self.w['analytic_grad'][dataset] = self.lr_analytic_grad()

            # with gradient descent using keras
            print(dataset + ", KERAS")
            self.w['keras'][dataset] = self.lr_keras()

            self.print_output(dataset)

    def set_parameters(self, dataset):
        """Set parameters for different datasets"""
        if dataset == 'breast_cancer':
            self.keras_input_dim = 30
            self.weight_vector_size = 31
            self.keras_batch_size = 426
            self.keras_learning_rate = 0.01
            self.keras_epochs = 1000
            self.ep = 0.01  # learning rate for analytic gradient
            self.iterations = 1000

        elif dataset == 'MNIST':
            self.keras_batch_size = 11085
            self.keras_input_dim = 784
            self.weight_vector_size = 785
            self.keras_learning_rate = 0.01
            self.ep = 0.01  # learning rate for analytic gradient
            self.keras_epochs = 500
            self.iterations = 500

    def get_dataset(self, dataset):
        if dataset == 'breast_cancer':
            self.X_train, self.y_train, self.X_test, self.y_test, self.X_train_keras, self.X_test_keras = self.create_breast_cancer_dataset()
        elif dataset == 'MNIST':
            self.X_train, self.y_train, self.X_test, self.y_test, self.X_train_keras, self.X_test_keras = self.create_MNIST_dataset()

    def create_breast_cancer_dataset(self):
        cancer = load_breast_cancer()

        X = cancer.data
        y = cancer.target

        # split up data set into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        # reshape label vectors and cast as floats
        y_train = np.reshape(y_train, (y_train.shape[0], 1)).astype(float)
        y_test = np.reshape(y_test, (y_test.shape[0], 1)).astype(float)

        # get mean and standard deviation of each feature in the training set
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)

        # subtract mean of each feature and divide by standard deviation
        X_train -= X_mean
        X_test -= X_mean
        X_train /= X_std
        X_test /= X_std

        # these don't have an extra column of 1s
        X_train_keras = X_train
        X_test_keras = X_test

        # put in column of 1s for bias variable
        X_train = matrix(np.hstack((np.ones((X_train.shape[0], 1)), X_train)))
        X_test = matrix(np.hstack((np.ones((X_test.shape[0], 1)), X_test)))

        return X_train, y_train, X_test, y_test, X_train_keras, X_test_keras

    def create_MNIST_dataset(self):
        mnist = fetch_mldata('MNIST original')
        X = mnist.data.astype(float)

        # reshape label vector so it has a second dimension of 1
        y = np.reshape(mnist.target, (mnist.target.shape[0],1))

        # put examples and labels into the same array
        examples_and_labels = np.hstack((X, y))

        # get all examples from two classes
        all_0s = X[np.where(examples_and_labels[:,-1] == 0)]
        all_1s = X[np.where(examples_and_labels[:, -1] == 1)]

        # get label vectors for the two classes
        y_0s = y[np.where(y == 0)]
        y_0s = np.reshape(y_0s, (y_0s.shape[0], 1))
        y_1s = y[np.where(y == 1)]
        y_1s = np.reshape(y_1s, (y_1s.shape[0], 1))

        # combine examples and labels from the two classes
        X = np.vstack((all_0s, all_1s))
        y = np.vstack((y_0s, y_1s))

        # split up data set into training and test sets and shuffle
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        # scale training and test data to unit norm
        X_train = normalize(X_train)
        X_test = normalize(X_test)

        # these don't have an extra column of 1s
        X_train_keras = X_train
        X_test_keras = X_test

        # put in column of 1s for bias variable
        X_train = matrix(np.hstack((np.ones((X_train.shape[0], 1)), X_train)))
        X_test = matrix(np.hstack((np.ones((X_test.shape[0], 1)), X_test)))

        """
        # test some images to make sure they are labeled correctly
        from PIL import Image

        w, h = 28, 28
        for i in [12,234,121,66,788,3444]:
            print X_train_keras[0,:].shape
            data = np.reshape(X_train_keras[i,:],(28,28))
            print "LABEL", y_train[i]
            img = Image.fromarray(data, 'L')
            img.show()
            stop = raw_input("enter something to continue")
        """

        # print X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_train_keras.shape, X_test_keras.shape
        return X_train, y_train, X_test, y_test, X_train_keras, X_test_keras

    def sigmoid(self, x):
        try:
            exponential = math.exp(-x)  # np.exp(-x)
        except OverflowError:
            print("OVERFLOW ERROR")
        return_val = 1.0 / (1.0 + exponential)
        return return_val

    # gets the predicted class if the predicted probability output by the logistic regression model is p
    def get_class(self, p):
        if p >= 0.5:
            return 1
        else:
            return 0

    # calculates the error rate of a classifier with weights w on a data set (X,y)
    def error_rate(self, w, X, y):
        error_rate = np.sum(
            np.absolute((self.elementwise_get_class(self.elementwise_sig(X * w))) - y)) / y.shape[0]
        return error_rate

    # calculates the accuracy of a classifier with weights w on a data set (X,y)
    def accuracy(self, w, X, y):
        accuracy = 1 - (
        np.sum(np.absolute((self.elementwise_get_class(self.elementwise_sig(X * w))) - y)) /
        y.shape[0])
        return accuracy

    def print_output(self, dataset):
        print("")
        print("DATASET:", dataset)

        print("Keras")
        #print "Weights:", self.w_keras
        print("Training Accuracy:", self.accuracy(self.w['keras'][dataset], self.X_train, self.y_train))
        print("Test Accuracy:", self.accuracy(self.w['keras'][dataset], self.X_test, self.y_test))
        print("")

        print("Analytic Gradient")
        #print "Weights:", self.w_analytic_grad
        print("Training Accuracy:", self.accuracy(self.w['analytic_grad'][dataset], self.X_train, self.y_train))
        print("Test Accuracy:", self.accuracy(self.w['analytic_grad'][dataset], self.X_test, self.y_test))
        print("")

    # computes the gradient of MSE_train at w analytically
    def analytic_grad(self, w):
        result = self.X_train.T * (self.elementwise_sig(self.X_train * w) - self.y_train)
        return result

    # runs gradient descent using the analytic formula for the gradient
    def lr_analytic_grad(self):
        # randomly initialize weights
        w = matrix(np.random.sample((self.weight_vector_size, 1)))

        # print performance with just random weights
        test_accuracy = self.accuracy(matrix(w).reshape((self.weight_vector_size, 1)), self.X_test, self.y_test)
        train_accuracy = self.accuracy(matrix(w).reshape((self.weight_vector_size, 1)), self.X_train, self.y_train)

        print("WITH RANDOM WEIGHTS:")
        print("TRAIN", train_accuracy)
        print("TEST", test_accuracy)
        print("")
        print("START TRAINING:")
        print("")

        count = 0

        while (True):
            step = self.ep * self.analytic_grad(w)
            w = w - step

            test_accuracy = self.accuracy(matrix(w).reshape((self.weight_vector_size,1)), self.X_test, self.y_test)
            train_accuracy = self.accuracy(matrix(w).reshape((self.weight_vector_size,1)), self.X_train, self.y_train)

            # print output every 100 steps
            if count % 100 == 0:
                print("TRAIN", train_accuracy)
                print("TEST", test_accuracy)
                print("")

            if count == self.iterations:
                break

            count += 1

        return matrix(list(w.flat)).reshape((self.weight_vector_size,1))

    # runs SGD using keras
    def lr_keras(self):
        model = Sequential()
        model.add(Dense(1, input_dim=self.keras_input_dim, activation='sigmoid'))
        sgd = SGD(lr=self.keras_learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_crossentropy', 'accuracy'])
        model.fit(self.X_train_keras, self.y_train, nb_epoch=self.keras_epochs, batch_size=self.keras_batch_size, verbose=2)
        weights = model.get_weights()

        return matrix([float(weights[1])] + [float(weights[0][i]) for i in range(len(weights[0]))]).reshape((self.keras_input_dim+1, 1))

lg = LogisticRegression(['MNIST', 'breast_cancer'])
lg.run()

