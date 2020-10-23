'''This module contains the 'IrisDataset' class.

Author: Julian Krygel, 20062527
'''

import numpy as np 


def return_class(data):
    '''Simple integer-tuple to label mapping. Returns a string containing the label of the data point passed in.'''
    labels = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
    for i in range(len(data)):
        if data[i] == 1:
            return labels[i]

# Define a function to represent the overall classification system.
def classify(dataset, classifier_1, classifier_2):
    '''Define the classification system.

    Define the classification system using two perceptrons. This first checks if each data point belongs to either Iris-setosa or Iris-virginica, and if it belongs to neither it is
    inferred to be a member of Iris-versicolour.

    Args:
        dataset: iris testing data, must be of class IrisDataset
        classifier_1: First perceptron, classifies Iris-setosa vs. non-Iris-setosa
        classifier_2: Second perceptron, classifies Iris-virginica vs. non-Iris-virginica
    Returns:
        results: A list of 2-element tuples, with the 1st element containing a point's predicted class and the 2nd containing its actual class
        errors_loc: A list of indices that identify incorrectly classified points. These points have a value of 1, while correctly classified points have a value of 0.
        num_errors: Total (int) number of misclassifications
    '''
    results = []
    errors_loc = np.zeros(dataset.labels.shape[0])
    num_errors = 0
    for i, feature in enumerate(dataset.data):
        prediction_1 = classifier_1.classify(feature)
        actual = return_class(dataset.labels[i]) 

        if prediction_1:
            # This instance is predicted to be of type Iris-setosa
            results.append(('Iris-setosa',actual))
            if not dataset.labels[i][0]:
                # Error
                errors_loc[i] = 1
                num_errors += 1
        else:
            prediction_2 = classifier_2.classify(feature)
            if prediction_2:
                # This instance is predicted to be of type Iris-virginica
                results.append(('Iris-virginica',actual))
                if not dataset.labels[i][1]:
                    # Error
                    errors_loc[i] = 1
                    num_errors += 1
            else:
                # This instance is predicted to be of type Iris-versicolor
                results.append(('Iris-versicolour',actual))
                if not dataset.labels[i][2]:
                    # Error
                    errors_loc[i] = 1
                    num_errors += 1
    return results, errors_loc, num_errors

class IrisDataset():
    '''Data class used to process and store contents of iris_test.txt and iris_train.txt/ 

    Attributes:
        data: An (8xn) numpy array storing iris data. Columns 0, 1, 2, and 4 contain sepal length,
            sepal width, petal length, and petal width respectively. Columns 5, 6, and 7 comprise a
            one-hot encoding scheme to represent the three categorical labels in the iris data:
            Column 5 represents Iris-setosa, 6 Iris-versicolor, and 7 Iris-virginica.
    '''
    def __init__(self, txt_path):
        '''Initialize class with path to .txt file containing data.'''
        self.data = None
        self.labels = None
        self.__import_data(txt_path)

    def __import_data(self, txt_path):
        '''Handle importing data, populate self.data and self.labels.'''

        data = np.genfromtxt(txt_path, delimiter=',',usecols=(0,1,2,3),dtype=None)
        cat_labels = np.genfromtxt(txt_path, delimiter=',',usecols=4,dtype=str)
        
        label_map = {
            'Iris-setosa': 0,
            'Iris-virginica': 1,
            'Iris-versicolor': 2
        }

        # If a data point belongs to a class, its values in that corresponding column is 1, else 0.
        labels = np.zeros((len(cat_labels),3))

        # The label representations are: Iris-setosa: (1,0,0); Iris-virginica: (0,1,0); Iris-versicolor: (0,0,1).
        for i, label in enumerate(cat_labels):
            index = label_map[label]
            labels[i,index] = 1
        
        # Make sure to add column of 1's to the training data to act as the bias input.
        bias = np.ones((data.shape[0],1))
        data = np.concatenate([bias,data], axis=1)

        self.data = data
        self.labels = labels
