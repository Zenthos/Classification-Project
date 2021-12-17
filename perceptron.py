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
        for label in legalLabels:
            self.weights[label] = 0


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
                    self.weights[prediction] = np.subtract(self.weights[prediction], datum.getPixels())
                    # Increase the weights of the correct label
                    self.weights[correct] = np.add(self.weights[correct], datum.getPixels())
            
            print("Iteration %d - Wrong prediction count: %d" % (iteration, wrongPredictionCount))


    """
    Creates a list of probabilities, based on the sums of the vectors
    """
    def softmax(self, x): 
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


    """
    Classifies each datum and attempts to predict which label matches the most
    """
    def predict(self, datum):
        # Calculate the vector sums of every label
        sums = []
        for label in self.legalLabels:
            vector = np.dot(self.weights[label], datum.getPixels())
            vectorSum = np.sum(vector)
            sums.append(vectorSum)

        # Calculate the probabilities of each label
        probabilities = self.softmax(sums)

        # Choose the label with the highest probability
        return np.argmax(probabilities)


# Training with 5000 digits
if __name__ == "__main__": 
    # Parses the datafiles into a list of Datum Objects, each holding a np.2d array size 28x28
    trainingData = dp.loadDataFile("digitdata/trainingimages", 5000, 28, 28)
    trainingLabels = dp.loadLabelFile("digitdata/traininglabels", 5000)

    # Needs to classify handwritten digits 0-9
    test = PerceptronClassifier(range(10), 3)
    test.train(trainingData, trainingLabels)
