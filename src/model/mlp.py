
import numpy as np

from util.loss_functions import BinaryCrossEntropyError
from util.loss_functions import SumSquaredError
from util.loss_functions import MeanSquaredError
from util.loss_functions import DifferentError
from util.loss_functions import AbsoluteError
from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'cee':
            self.loss = CrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str(loss))

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10, 
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _get_output(self):
        return self._get_output_layer().outp

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer
        Returns
        -------
        outp: ndarray
            The output of the last layer.

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        nextInput = inp
        for layerIndex in range(len(self.layers)):
            layer = self.layers[layerIndex]
            # Output of this layer is the input of the next layer:
            nextInput = layer.forward(nextInput)
            # Add bias "1" at the beginning (except for last layer):
            if layerIndex < (len(self.layers) - 1):
                nextInput = np.insert(nextInput, 0, 1)

        # Return the output of the last layer:
        return nextInput

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the error
        """
        return self.loss.calculateDerivative(target, self._get_output())
    
    def _update_weights(self, learningRate, label):
        """
        Update the weights of the layers by propagating back the error
        """
        # Backpropagation of error:
        # To calculate the derivatives, we iterate over the layers in reverse order:
        # In case of the output layer, next_weights is array of 1
        # and next_derivatives - the derivative of the error will be the errors
        target = np.zeros(self._get_output_layer().nOut)
        target[label] = 1
        next_derivatives = self._compute_error(target)
        # this produces a result equivalent to using the identity
        next_weights = 1.0
        for layer in reversed(self.layers):
            # Compute the derivatives:
            next_derivatives = layer.computeDerivative(next_derivatives, next_weights)
            # Remove bias from weights, so it matches the output size of the next layer:
            next_weights = layer.weights[1:,:].T

        # Update the weights:
        for layer in self.layers:
            layer.updateWeights(learningRate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        #lastAccuracy = 0.0
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            # Train the network:
            self._train_one_epoch()
            # Determine accuracy by evaluating the validation set:
            accuracy = accuracy_score(self.validationSet.label, self.evaluate(self.validationSet))
            # Record the performance of each epoch for later usages
            # e.g. plotting, reporting..
            self.performances.append(accuracy)

            if verbose:
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

            #if accuracy <= lastAccuracy:
                # Reached stop criteria to prevent overfitting:
                #print("Reached stop criteria")
            #    break
            # Else, update last accuracy:
            #lastAccuracy = accuracy

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        for input, label in zip(self.trainingSet.input, self.trainingSet.label):
            # Compute the network output via feed forward:
            self._feed_forward(input)
            # Backpropagate the error and update the weights
            self._update_weights(self.learningRate, label)

    def classify(self, test_instance):
        """Classify a single instance.

        Parameters
        ----------
        test_instance : list of floats

        Returns
        -------
        int :
            The recognized digit (0-9).
        """
        # Compute the network output via feed forward:
        output = self._feed_forward(test_instance)
        return np.argmax(output)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
