from __future__ import division
import numpy as np
from numpy import matrix
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import normalize
from keras.datasets import mnist, cifar10, cifar100

np.random.seed(1000)

class MulticlassClassification:

    def __init__(self, datasets):
        # a python list containing the names of datasets run multiclass classification on
        self.datasets = datasets

        # use to store trained models
        self.trained_models = {}

    def run(self):
        """Do multiclass classification on MNIST using a CNN"""
        for dataset in self.datasets:
            self.get_dataset(dataset)
            self.set_parameters(dataset)

            print(dataset + ", KERAS")
            self.trained_models[dataset]= self.mcc_keras()

            self.print_output(dataset)

    def set_parameters(self, dataset):
        """Set parameters for different datasets"""
        if dataset == 'MNIST':
            self.keras_batch_size = 30
            self.keras_input_shape = (28, 28, 1)
            self.keras_learning_rate = 0.1
            self.keras_epochs = 5

    def get_dataset(self, dataset):
        if dataset == 'MNIST':
            self.X_train_keras, self.X_test_keras, self.y_train_keras, self.y_test_keras = self.create_MNIST_dataset()

    def create_MNIST_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # scale training and test data to unit norm
        X_train = np.reshape(X_train, (X_train.shape[0], 784))
        X_test = np.reshape(X_test, (X_test.shape[0], 784))
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

        # these don't have an extra column of 1s
        X_train_keras = X_train
        X_test_keras = X_test

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
        return X_train_keras, X_test_keras, y_train_keras, y_test_keras

    def print_output(self, dataset):
        print("")
        print("DATASET:", dataset)

        print("Keras")
        # print "Weights:", self.w_keras
        print("")
        scores = self.trained_models[dataset].evaluate(self.X_train_keras, self.y_train_keras, verbose=0)
        for metric, value in zip(scores, self.trained_models[dataset].metrics_names):
            print("TRAINING", metric, value)
        print("")
        scores = self.trained_models[dataset].evaluate(self.X_test_keras, self.y_test_keras, verbose=0)
        for metric, value in zip(scores, self.trained_models[dataset].metrics_names):
            print("TESTING", metric, value)

    # runs SGD using keras
    def mcc_keras(self):
        model = Sequential()
        model.add(Convolution2D(60, 5, 5, border_mode='valid', input_shape=self.keras_input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid')) # pool_size=(2,2) cuts width and height of feature map in half
        model.add(Convolution2D(60, 5, 5, border_mode='valid', input_shape=model.layers[-1].output_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))  # pool_size=(2,2) cuts width and height of feature map in half
        model.add(Flatten())
        model.add(Dense(300, input_shape=model.layers[-1].output_shape, activation='relu'))
        model.add(Dense(150, input_shape=model.layers[-1].output_shape, activation='relu'))
        for layer in model.layers:
            print(layer.name)
            print(layer.output_shape)
        model.add(Dense(10, input_shape=model.layers[-1].output_shape, activation='softmax'))

        sgd = SGD(lr=self.keras_learning_rate, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_crossentropy', 'accuracy'])
        model.fit(self.X_train_keras, self.y_train_keras, nb_epoch=self.keras_epochs, batch_size=self.keras_batch_size)
        return model

lg = MulticlassClassification(['MNIST'])
lg.run()

