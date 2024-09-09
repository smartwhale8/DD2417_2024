from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.01  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 100 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        z = np.dot(self.theta, self.x[datapoint])
        prob = self.sigmoid(z)
        return prob if label == 1 else (1-prob)


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        #compute the predicted probabilities for each data point using the conditional_probabilities
        probabilities = np.array([self.conditional_prob(1, i) for i in range(self.DATAPOINTS)])

        #Calculate the errors as the difference between predicted probabilities and the actual labels
        errors = probabilities - self.y

        #Compute the gradient for the weight vectors
        gradient = np.dot(self.x.T, errors) / self.DATAPOINTS
        self.gradient = gradient

        return self.gradient


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        x_minibatch = self.x[minibatch]
        y_minibatch = self.y[minibatch]

        z = np.dot(x_minibatch, self.theta)
        probabilities = self.sigmoid(z)

        errors = probabilities - y_minibatch
        gradient = np.dot(x_minibatch.T, errors) / len(minibatch)
        self.gradient = gradient

        return self.gradient

    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """
        prediction = self.conditional_prob(1, datapoint)        
        error = prediction - self.y[datapoint]

        #computation of gradient using vectorized operation, avoid for loop over the features
        gradient = self.x[datapoint] * error 
        self.gradient = gradient

        #Optional return the value; self.gradient has already been updated
        return self.gradient

    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        Here, the model parameters are updated using the gradient computed from a 
        randomly selected single data point rather than the entire dataset.
        Approach is often faster, and can converge more quickly for larger datasets
        ,although it introduced more variability in the learning process.        
        """
        self.init_plot(self.FEATURES)

        # Random permutation of indices to go through each datapoint exactly once per epoch
        indices = np.arange(self.DATAPOINTS)

        #iterate until convergence of max iterations reached
        for iteration in range(self.MAX_ITERATIONS):

            #Shuffle indices to ensure randomnes
            np.random.shuffle(indices)
            
            for idx in indices:
                self.compute_gradient(idx)
                self.theta -= self.LEARNING_RATE * self.gradient

                #update the plot with the new values (every few iters, for performance)
                if iteration % 10 == 0 and idx == indices[-1]: #Update the plot at the end of each eopoc:
                    print(f"Working with norm of gradient {np.linalg.norm(self.gradient)} at {iteration} iteration")
                    self.update_plot(np.linalg.norm(self.gradient))

            #Early stopping criteria:
            #Check if the norm of the gradient is less than the convergence margin
            if np.linalg.norm(self.gradient) < self.CONVERGENCE_MARGIN:
                print(f"Convergence reacher after {iteration +1 } iterations.")
                break
        self.print_result()

    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        converged = False #initialize

        # Create an array of indices for the datapoints
        indices = np.arange(self.DATAPOINTS)

        for iteration in range(self.MAX_ITERATIONS):
            np.random.shuffle(indices) #shuffle the indices to randomizes the mini-batch selection

            # Split indices into mini-batches
            """Return an object that produces a sequence of integers from
              start (inclusive) to stop (exclusive) by step"""
            minibatches = [indices[k:k + self.MINIBATCH_SIZE] for k in range(0, self.DATAPOINTS, self.MINIBATCH_SIZE)]

            for minibatch in minibatches:
                # Compute the gradient for the minibatch
                self.compute_gradient_minibatch(minibatch)
                self.theta -= self.LEARNING_RATE * self.gradient #Update the model parameters

                #update the plot with the new values
                if iteration % 10 == 0: #Update the plot 10 iterations for performance
                    #print(f"Working with norm of gradient {np.linalg.norm(self.gradient)} at {iteration} iteration")
                    self.update_plot(np.linalg.norm(self.gradient))

                #check if the norm of the gradient is less than the convergence margin to stop early
                if np.linalg.norm(self.gradient) < self.CONVERGENCE_MARGIN:
                    print(f"Convergence reached after {iteration + 1} iterations.") 
                    converged = True
                    break
            
            if converged:
                break

        self.print_result() #Print the final results after training



    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        # Initialise iteration to 0. Will be used for tracking
        itr = 0 
        while True:
            itr += 1

            self.compute_gradient_for_all()

            for k in range(self.FEATURES):
                self.theta[k] -= self.LEARNING_RATE * self.gradient[k]
            
            # For tracking purposes
            # Note: Convergence means sum of square of gradient[k] is smaller than some convergence margin
            if itr == 1 or itr % 10 == 0:
                print("Iter: {} , Sum of square of Gradient: {} ".format(itr, np.sum(np.square(self.gradient))))
                self.update_plot(np.sum(np.square(self.gradient)))

            # Terminating condition
            if np.sum(np.square(self.gradient)) < self.CONVERGENCE_MARGIN:
                print("At termination, Iter: {} , Sum of Square of Gradient: {}:".format(itr,np.sum(np.square(self.gradient))))
                break


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()