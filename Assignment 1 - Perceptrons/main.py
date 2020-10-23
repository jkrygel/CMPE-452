'''
Author: Julian Krygel, 20062527
'''

import sys
import matplotlib.pyplot as plt
from perceptron import Perceptron, PocketPerceptron
from utils import IrisDataset, return_class, classify

def main() :
    # Import training and testing data
    train_1 = IrisDataset('iris_train.txt')
    train_2 = IrisDataset('iris_train.txt')
    train = IrisDataset('iris_train.txt')
    test = IrisDataset('iris_test.txt')

    # This network consists of two output neurons classifying 3 labels (the 3rd label is inferred by not belonging to either the 1st or 2nd label). Each neuron is trained seperately,
    # so split up labels for training each one.
    train_1.labels = train_1.labels[:,0]
    train_2.labels = train_2.labels[:,1]

    # For the 2nd classifier (Iris-versicolor vs Iris-virginica), omit Iris-setosa data. For some reason, the 2nd classifier does not train well at all with the inclusion of Iris-setosa data.
    train_2.data = train_2.data[40:]
    train_2.labels = train_2.labels[40:]
    
    # Now, train both classifiers.
    classifier_1 = Perceptron()
    classifier_1.train(train_1.data, train_1.labels)
    classifier_2 = Perceptron()
    classifier_2.train(train_2.data, train_2.labels)  
   
    # The classify function in utils.py defines the overall architecture of the classification system, and returns an array of 2 element tuples containing the predicted and actual labels of
    # every point in the test set.
    results, errors_loc1, num_errors1 = classify(test, classifier_1, classifier_2)
    # Print out the results to a file. As you can see in the output, there are 3 total missclassifications using the test data.
    with open('results1_test.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('Prediction', '\t', 'Actual')
        for _, value in enumerate(results):
            a, b = value
            print(a, '\t', b)
        sys.stdout = original_stdout # Reset the standard output to its original value
    
    # Run the classifier on the training data just to identify the linearly inseparable pointa
    results, errors_loc1, num_errors1 = classify(train, classifier_1, classifier_2)
    with open('results1_train.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('Prediction', '\t', 'Actual')
        for _, value in enumerate(results):
            a, b = value
            print(a, '\t', b)
        sys.stdout = original_stdout # Reset the standard output to its original value


    # Next, let's train both classifiers again but with the pocket algorithm.
    pclassifier_1 = PocketPerceptron()
    pclassifier_1.train(train_1)
    pclassifier_2 = PocketPerceptron()
    pclassifier_2.train(train_2) 
    
    # Let's feed the test data through the pocket algorithm classifier
    results, errors_loc2, num_errors2 = classify(test, pclassifier_1, pclassifier_2)
    # Print out the results to a file.
    with open('results2_test.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('Prediction', '\t', 'Actual')
        for _, value in enumerate(results):
            a, b = value
            print(a, '\t', b)
        sys.stdout = original_stdout # Reset the standard output to its original value
    
    # Run the classifier on the training data just to identify the linearly inseparable pointa
    results, errors_loc1, num_errors1 = classify(train, pclassifier_1, pclassifier_2)
    with open('results2_train.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print('Prediction', '\t', 'Actual')
        for _, value in enumerate(results):
            a, b = value
            print(a, '\t', b)
        sys.stdout = original_stdout # Reset the standard output to its original value
    


if __name__ == "__main__":
    original_stdout = sys.stdout # Save a reference to the original standard output
    main()