from __future__ import division
import numpy as np
from numpy import matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from keras.utils import np_utils
from sklearn.preprocessing import normalize

np.random.seed(1000)

class MulticlassClassification:

    def __init__(self, datasets):
        # a python list containing the names of datasets run logistic regression on
        self.datasets = datasets

        # use to store weights found after optimization
        self.m = {}
        self.m['keras'] = {}

    def run(self):
        """Run multiclass classification with a feed forward network"""
        for dataset in self.datasets:
            self.get_dataset(dataset)
            self.set_parameters(dataset)

            # with gradient descent using keras
            print(dataset + ", KERAS")
            self.m['keras'][dataset] = self.mcc_keras()

            self.print_output(dataset)

    def set_parameters(self, dataset):
        """Set parameters for different datasets"""
        if dataset == 'MNIST':
            self.keras_batch_size = 100
            self.keras_input_dim = 784
            self.weight_vector_size = 785
            self.keras_learning_rate = 0.05
            self.ep = 0.01  # learning rate for analytic gradient
            self.keras_epochs = 5
            self.iterations = 500

    def get_dataset(self, dataset):
        if dataset == 'MNIST':
            self.X_train, self._kerasy_train, self.X_test, self.y_test, self.X_train_keras, \
            self.X_test_keras, self.y_train_keras, self.y_test_keras = self.create_MNIST_dataset()

    def create_MNIST_dataset(self):
        mnist = fetch_mldata('MNIST original')
        X = mnist.data.astype(float)

        # reshape label vector so it has a second dimension of 1
        y = np.reshape(mnist.target, (mnist.target.shape[0],1))

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

        # convert labels to binary array
        y_train_keras = np_utils.to_categorical(y_train)
        y_test_keras = np_utils.to_categorical(y_test)

        # print X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_train_keras.shape, X_test_keras.shape
        return X_train, y_train, X_test, y_test, X_train_keras, X_test_keras, y_train_keras, y_test_keras

    def print_output(self, dataset):
        print("")
        print("DATASET:", dataset)

        print("Keras")
        print("")
        scores = self.m['keras'][dataset].evaluate(self.X_train_keras, self.y_train_keras, verbose=0)
        for metric, value in zip(scores, self.m['keras'][dataset].metrics_names):
            print("TRAINING", metric, value)
        print("")
        scores = self.m['keras'][dataset].evaluate(self.X_test_keras, self.y_test_keras, verbose=0)
        for metric, value in zip(scores, self.m['keras'][dataset].metrics_names):
            print("TESTING", metric, value)

    # runs SGD using keras
    def mcc_keras(self):
        model = Sequential()
        model.add(Dense(500, input_dim=self.keras_input_dim, activation='relu'))
        model.add(Dense(500, input_dim=5, activation='relu'))
        model.add(Dense(500, input_dim=5, activation='relu'))
        model.add(Dense(10, input_dim=5, activation='sigmoid'))
        sgd = SGD(lr=self.keras_learning_rate, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_crossentropy', 'accuracy'])
        model.fit(self.X_train_keras, self.y_train_keras, nb_epoch=self.keras_epochs, batch_size=self.keras_batch_size)
        return model

lg = MulticlassClassification(['MNIST'])
lg.run()




