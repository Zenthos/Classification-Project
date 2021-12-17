import dataParser as dp
import numpy as np


"""
Naive Bayes classifier.
"""
class NaiveBayesClassifier:
    def __init__(self, legalLabels, maxIterations):
        self.legalLabels = legalLabels
        self.type = "Naive Bayes"
        self.frequencies = {}
        self.summaries = {}
        self.likelihoods = {}
        self.priors = {}
        for label in legalLabels:
            self.priors[label] = 0
            self.frequencies[label] = 0


    """
    Trains the model by calculating probabilities
    """
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        datumWidth = trainingData[0].width
        datumHeight = trainingData[0].height

        # Initialize the summaries with the value of 1, to ensure every value pixel is accounted for
        for label in self.legalLabels:
            self.likelihoods[label] = np.zeros((datumWidth, datumHeight))
            self.summaries[label] = np.ones((datumWidth, datumHeight))

        # Sum up how many times a label is seen and how many times a pixel appears in a certain spot for a given label
        for index, datum in enumerate(trainingData):
            label = trainingLabels[index]
            self.frequencies[label] = self.frequencies[label] + 1
            self.summaries[label] = np.add(self.summaries[label], datum.getFeatures())

        # Calculated the probability of a label
        for label in self.priors:
            self.priors[label] = self.frequencies[label] / (len(trainingLabels) + len(self.legalLabels))

        # Calculate the probabilities of a pixel given a label
        for label in self.likelihoods:
            self.likelihoods[label] = np.divide(self.summaries[label], np.full((datumWidth, datumHeight), len(trainingLabels)))

        print("Finished calculating priors and likelihoods...")

    
    """
    Classifies each datum and attempts to predict which label matches the most
    """
    def predict(self, datum):
        guesses = []

        # Multiply the probabilities with the features to get the posterior of each label
        for label in self.legalLabels:
            vector = np.dot(datum.getFeatures(), self.likelihoods[label])
            posterior = np.sum(vector)
            guesses.append(posterior)

        # The posterior with the highest value is the prediction
        return np.argmax(guesses)


    """
    Takes in a list of datums and predicts a value for each one
    """
    def classify(self, testData):
        return [self.predict(datum) for datum in testData]


# Training with 5000 digits
if __name__ == "__main__": 
    # Parses the datafiles into a list of Datum Objects, each holding a np.2d array size 28x28
    trainingData = dp.loadDataFile("digitdata/trainingimages", 5000, 28, 28)
    trainingLabels = dp.loadLabelFile("digitdata/traininglabels", 5000)

    # Needs to classify handwritten digits 0-9
    test = NaiveBayesClassifier(range(10), 0)
    test.train(trainingData, trainingLabels, [], [])
    print(test.predict(trainingData[0]))
