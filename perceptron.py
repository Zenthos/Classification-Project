import dataParser as dp
import numpy as np


"""
Perceptron classifier.
"""
class PerceptronClassifier:
    def __init__(self, legalLabels, maxIterations):
        self.legalLabels = legalLabels
        self.maxIterations = maxIterations
        self.type = "Perceptron"
        self.weights = {}
        self.bias = {}
        self.alpha = 0.1
        for label in legalLabels:
            self.weights[label] = 0
            self.bias[label] = 0

    """
    Trains by updating the weights of each label based on whether the prediction matches the label
    """
    def train(self, trainingData, trainingLabels):
        # Set the weight dimensions by figuring out the size of a datum
        for label in self.weights:
            self.weights[label] = np.zeros((trainingData[0].width, trainingData[0].height))

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
                    self.bias[prediction] -= self.alpha
                    self.weights[prediction] -= self.alpha * datum.getFeatures()
                    # Increase the weights of the correct label
                    self.bias[correct] += self.alpha
                    self.weights[correct] += self.alpha * datum.getFeatures()

            print("Iteration %d - Wrong prediction count: %d" % (iteration, wrongPredictionCount))


    """
    Classifies each datum and attempts to predict which label matches the most
    """
    def predict(self, datum):
        # Calculate the activation sums of every label
        activations = []
        for label in self.legalLabels:
            activation = np.dot(self.weights[label].flatten(), datum.getFeatures().flatten())
            activation += self.bias[label]
            activations.append(activation)

        # Choose the label with the highest probability
        return np.argmax(activations)


# Training with 5000 digits
if __name__ == "__main__":
    # Parses the datafiles into a list of Datum Objects, each holding a np.2d array size 28x28
    trainingData = dp.loadDataFile("digitdata/trainingimages", 5000, 28, 28)
    trainingLabels = dp.loadLabelFile("digitdata/traininglabels", 5000)

    # Needs to classify handwritten digits 0-9
    test = PerceptronClassifier(range(10), 3)
    test.train(trainingData, trainingLabels)