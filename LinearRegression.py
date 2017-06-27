from __future__ import division
import numpy as np
from numpy import matrix
import plotly.graph_objs as go
from plotly.offline import plot
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
np.random.seed(22)


class LinearRegression:

    def __init__(self):
        # create data set
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_train_keras, self.X_test_keras = self.create_dataset()

        # python list of points to graph
        self.x_points_train = list(self.X_train_keras.flat)
        self.y_points_train = list(self.y_train.flat)
        self.x_points_test = list(self.X_test_keras.flat)
        self.y_points_test = list(self.y_test.flat)

        # gradient descent hyper parameters
        self.learning_rate = 0.1  # learning rate
        self.steps = 1000  # number of steps to take in gradient descent for keras and analytic gradient
        self.steps_est = 5000  # number of steps to take in gradient descent when using the estimated gradient

        # use to store weights found after optimization
        self.w_normal = None
        self.w_est_grad = None
        self.w_analytic_grad = None
        self.w_keras = None

        # size of small movement (used to estimate gradient)
        self.h = 0.0000000000001

    def run(self):
        """Do linear regression 4 ways"""

        # with normal equations
        self.w_normal = self.lr_normal()

        # with gradient descent using estimated gradients
        self.w_est_grad = self.lr_est_grad()

        # with gradient descent using analytically computed gradients
        self.w_analytic_grad = self.lr_analytic_grad()

        # with gradient descent using Keras
        self.w_keras = self.lr_keras()

        # graph all lines on scatter plots with training data and test data
        self.graph_results()

        # print weights
        print(self.w_normal)
        print(self.w_est_grad)
        print(self.w_analytic_grad)
        print(self.w_keras)

    def lr_normal(self):
        return list(((self.X_train.T * self.X_train).I * (self.X_train.T * self.y_train)).flat)

    def lr_est_grad(self):
        # randomly initialze w
        w = matrix(np.random.sample((2, 1)))

        count = 0
        while (True):
            step = self.learning_rate * self.grad_est(self.MSE_train, w)
            w = w - step

            count += 1
            if count == self.steps_est:
                break

        return list(w.flat)

    def lr_analytic_grad(self):
        # randomly initialize w
        w = matrix(np.random.sample((2, 1)))
        count = 0
        while (True):
            step = self.learning_rate * self.analytic_grad(w)
            w = w - step

            count += 1
            if count == self.steps:
                break
        return list(w.flat)

    def lr_keras(self):
        model = Sequential()
        model.add(Dense(output_dim=1, input_dim=1))
        model.add(Activation('linear'))
        sgd = SGD(lr=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])
        model.fit(self.X_train_keras, self.y_train, nb_epoch=self.steps, batch_size=1000)
        return [float(model.get_weights()[1]),float(model.get_weights()[0])]

    def create_dataset(self):
        # get x values in the interval [0,1)
        X_train = matrix(np.reshape(np.random.sample(1000),(1000, 1)))
        X_test = matrix(np.reshape(np.random.sample(200),(200,1)))

        # sample y value from a line with equation 5 + 1.5x with normally distributed noise
        def get_y_coord(x):
            return np.random.normal(5 + 1.5*x, 0.5)

        coordinates = np.vectorize(get_y_coord)

        y_train = matrix(coordinates(X_train))
        y_test = matrix(coordinates(X_test))

        # these don't have an extra column of 1s
        X_train_keras = X_train
        X_test_keras = X_test

        # put in column of 1s for bias variable
        X_train = matrix(np.hstack((np.ones((1000, 1)), X_train)))
        X_test = matrix(np.hstack((np.ones((200, 1)), X_test)))

        return X_train, y_train, X_test, y_test, X_train_keras, X_test_keras

    # computes an estimate of grad(f) at w
    def grad_est(self, f, w):
        return matrix([
            [(f(w + matrix([[self.h], [0]])) - f(w)) / self.h],
            [(f(w + matrix([[0], [self.h]]))-f(w)) / self.h]
        ])

    # computes the gradient of MSE_train at w analytically
    def analytic_grad(self, w):
        result = (1/(2*1000)) * \
                 (2 * (self.X_train.T * self.X_train * w) - 2 *
                  (self.X_train.T * self.y_train))
        return result

    # computes the value of MSE_train at w
    def MSE_train(self, w):
        return (1 / 1000) * np.linalg.norm(self.X_train * w - self.y_train)

    def graph_results(self):
        min_train = min(self.x_points_train)
        max_train = max(self.x_points_train)
        min_test = min(self.x_points_test)
        max_test = min(self.x_points_test)

        # regression lines
        line1 = go.Scatter(
            x=[min_train, max_train],
            y=[(self.w_normal[0] + self.w_normal[1] * min_train), (self.w_normal[0] + self.w_normal[1] * max_train)],
            name='normal',
            mode='lines'
        )

        line2 = go.Scatter(
            x=[min_train, max_train],
            y=[(self.w_est_grad[0] + self.w_est_grad[1] * min_train), (self.w_est_grad[0] + self.w_est_grad[1] * max_train)],
            name='estimated gradient',
            mode='lines'
        )

        line3 = go.Scatter(
            x=[min_train, max_train],
            y=[(self.w_analytic_grad[0] + self.w_analytic_grad[1] * min_train), (self.w_analytic_grad[0] + self.w_analytic_grad[1] * max_train)],
            name='analytic gradient',
            mode='lines'
        )

        line4 = go.Scatter(
            x=[min_train, max_train],
            y=[(self.w_keras[0] + self.w_keras[1] * min_train), (self.w_keras[0] + self.w_keras[1] * max_train)],
            name='keras',
            mode='lines'
        )

        line5 = go.Scatter(
            x=[min_train, max_train],
            y=[(5 + 1.5 * min_train), (5 + 1.5 * max_train)],
            name='actual line',
            mode='lines'
        )

        # points
        training_points = go.Scatter(
            x=self.x_points_train,
            y=self.y_points_train,
            mode='markers'
        )
        test_points = go.Scatter(
            x=self.x_points_test,
            y=self.y_points_test,
            mode='markers'
        )
        # graph of all regression lines on training set scatter plot
        plot([line1, line2, line3, line4, line5, training_points], filename="lines_with_training_set.html")
        # graph of all regression lines on test set scatter plot
        plot([line1, line2, line3, line4, line5, test_points], filename="lines_with_test_set.html")

lg = LinearRegression()
lg.run()