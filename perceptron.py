import numpy as np


"""
Single Layer Perceptron classifier.
"""
class PerceptronClassifier:
    def __init__(self, legalLabels, maxIterations):
        self.legalLabels = legalLabels
        self.maxIterations = maxIterations
        self.type = "Perceptron"
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = 0


    """
    Trains by updating the weights of each label based on whether the prediction matches the label
    """
    def train(self, trainingData, trainingLabels):
        # Set the weight dimensions by figuring out the size of a datum
        for label in self.weights:
            self.weights[label] = np.zeros((trainingData[0].height, trainingData[0].width))

        # Begin training
        for iteration in range(self.maxIterations):
            wrongPredictionCount = 0
            for index, datum in enumerate(trainingData):
                # Make a guess
                prediction = self.predict(datum)
                correct = trainingLabels[index]

                # If prediction was wrong, readjust the weights
                if prediction != correct:
                    wrongPredictionCount += 1
                    # Decrease the weights of the prediction
                    self.weights[prediction] -= datum.getFeatures()
                    # Increase the weights of the correct label
                    self.weights[correct] += datum.getFeatures()

            print("Iteration %d - Wrong prediction count: %d" % (iteration, wrongPredictionCount))


    """
    Classifies each datum and attempts to predict which label matches the most
    """
    def predict(self, datum):
        # Calculate the activation sums of every label
        activations = []
        for label in self.legalLabels:
            activation = np.dot(self.weights[label].flatten(), datum.getFeatures().flatten())
            activations.append(activation)

        # Choose the label with the highest probability
        return np.argmax(activations)


    """
    Takes in a list of datums and predicts a value for each one
    """
    def classify(self, testData):
        return [self.predict(datum) for datum in testData]
