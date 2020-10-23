'''This module contains the 'Perceptron' class.

Author: Julian Krygel, 20062527
'''

import numpy as np
from copy import copy

class Perceptron():
    '''Implementation of the Perceptron architecture/algorithm (Rosenblatt, 1958).

    Attributes:
        weights: Learned perceptron weights
        l_rate: Learning rate
        epochs: Number of epochs to train over
        errors: Array containing cumulative error per epoch 
    '''
    def __init__(self):
        '''Inits Perceptron class.'''
        self.weights = None
        self.l_rate = None
        self.epochs = None
        self.errors = None

    def train(self, data, labels, l_rate=1, epochs=120):
        '''Train the perceptron. 

        Train using simple feedback learning. Iterates through every point in the training dataset once per epoch.

        Args:
            data: training data
            labels: training labels
        '''
        self.l_rate = l_rate
        self.epochs = epochs
        self.errors = []

        # Initialize all weights to 0
        self.weights = np.zeros(data.shape[1])
        
        for i in range(self.epochs):
            errors = 0

            for j, features in enumerate(data):
                output = self.classify(features)
                self.weights += self.l_rate*(labels[j] - output)*features
                if labels[j] != output:
                    # Weights are updated, increment error count
                    errors += 1
            self.errors.append(errors)
    
    def classify(self, data):
        '''Classify data using a threshold activation function.'''
        preds = np.dot(self.weights, data)
        return np.where(preds > 0.0, 1, 0)

class PocketPerceptron(Perceptron):
    '''Extension of the Perceptron class to utilize the 'pocket' algorithm (Gallant, 1990).
    
    Extended Attributes:
        run_length: The run length of the current self.weights.
    '''
    def __init__(self):
        super().__init__()
        self.run_length = 0
    
    def train(self, dataset, l_rate=1, epochs=140):
        '''Train the pocket algorithm perceptron. 

        Train using simple feedback learning and the 'pocket' algorithm. Iterates through every point in the training dataset once per epoch.

        Args:
            dataset: iris training dataset
        '''
        data = dataset.data
        labels = dataset.labels
        self.l_rate = l_rate
        self.epochs = epochs
        self.errors = []
        
        # Keep best weights and run length in 'pocket'
        best_w = np.zeros(data.shape[1])
        best_run = 0

        for i in range(self.epochs):
            errors = 0
            # Initialize weights to pocket weight, run length to 0
            self.weights = best_w
            self.run_length = 0
            
            for j, features in enumerate(data):
                output = self.classify(features)
                self.weights += self.l_rate*(labels[j] - output)*features

                if labels[j] != output:
                    # Weights are updated, increment error count and reset run length
                    errors += 1
                    self.run_length = 0
                else:
                    # Weights unchanged, run continues
                    self.run_length += 1
            
            if self.run_length > best_run:
                # If current run length is greater than run length of weights in the pocket, replace both th weights and run length in the pocket
                best_w = self.weights
                best_run = self.run_length

            self.errors.append(errors)
        
        # After training is concluded, replace weight and run_length attributes with those stored in the pocket
        self.weights = best_w
        self.run_length = best_run
