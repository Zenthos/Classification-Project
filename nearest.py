from operator import itemgetter
import numpy as np
import math


"""
K Nearest Neighbor Classifier.
"""
class KNearestNeighborClassifier:
    def __init__(self, legalLabels, maxIterations):
        self.legalLabels = legalLabels
        self.type = "K-Nearest Neighbor"
        self.k = 3


    """
    No actual training occurs, the algorithm works during classification
    """
    def train(self, trainingData, trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels

        # Cap the k value to the size of the training data
        self.k = len(self.trainingData) if self.k > len(self.trainingData) else self.k


    """
    Calculates the test image's distance from the trainingImage
    """
    def distance(self, testImage): 
        distances = []

        # The distance is determined by using the euclidean metric
        for index, trainingImage in enumerate(self.trainingData):
            v = np.sum((testImage.getFeatures() - trainingImage.getFeatures()) ** 2)
            distances.append((self.trainingLabels[index], math.sqrt(v)))

        return distances

    """
    Classifies each datum and attempts to predict which label matches the most
    """
    def predict(self, datum):
        # Calculate the distances of each trainingImage with the datum we wanna predict
        distances = self.distance(datum)

        # Sorts the tuples by the smallest distance first
        sortedDistances = sorted(distances, key=itemgetter(1))

        # Initialize dict to keep track of number of labels appeared in k-count
        occurrences = {}
        for label in self.legalLabels:
            occurrences[label] = 0

        # Count the number of occurrences of a label, k number of times
        for k in range(self.k):
            occurrences[sortedDistances[k][0]] += 1

        # The label that appeared the most times in the k-dict is the prediction
        return max(occurrences, key=occurrences.get)
    

    """
    Takes in a list of datums and predicts a value for each one
    """
    def classify(self, datums):
        return [self.predict(datum) for datum in datums]
