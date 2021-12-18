import math


"""
Naive Bayes classifier.
"""
class NaiveBayesClassifier:
    def __init__(self, legalLabels, maxIterations):
        self.legalLabels = legalLabels
        self.type = "Naive Bayes"
        self.trainingSet = {}
        self.likelihoods = {}
        self.priors = {}
        for label in legalLabels:
            self.trainingSet[label] = []


    """
    Calculates P(Label)
    """
    def calcPriorProbabilities(self, trainingData):
        # Compute prior probability of each label
        for label in self.legalLabels:
            self.priors[label] = (len(self.trainingSet[label]) + 1) / (len(trainingData) + len(self.legalLabels))


    """
    Calculates P(Pixel > 0 | Label)
    """
    def calcPixelProbabilityGivenLabel(self, label):
        gridProb = [[0 for _ in range(self.DATUM_WIDTH)] for _ in range(self.DATUM_HEIGHT)]

        # Calculates the probability of every single pixel
        for row in range(self.DATUM_HEIGHT):
            for col in range(self.DATUM_WIDTH):
                chars = 0
                blank = 0

                # Count the number of 1's and 0's inside the features array
                for sample in self.trainingSet[label]:
                    if sample.getFeature(row, col) > 0:
                        chars += 1
                    else:
                        blank += 1

                # The CPT of a single pixel [P(>0), P'(>0)]
                probability = []

                # Apply smoothing to ensure the probability for every pixel is > 0
                probability.append((blank + 1)/ (float(len(self.trainingSet[label])) + 2))
                probability.append((chars + 1)/ (float(len(self.trainingSet[label])) + 2))

                gridProb[row][col] = probability

        # Add 3d matrix to dictionary with label as key
        self.likelihoods[label] = gridProb


    """
    Trains the model by calculating probabilities
    """
    def train(self, trainingData, trainingLabels):
        # Set Datum Sizes
        self.DATUM_WIDTH = trainingData[0].width
        self.DATUM_HEIGHT = trainingData[0].height

        # Partition the trainingData based on the labels
        for index, trainingImage in enumerate(trainingData):
            self.trainingSet[trainingLabels[index]].append(trainingImage)

        # Calculate Priors
        self.calcPriorProbabilities(trainingData)

        # Calculate Pixel Likelihoods given a label
        for label in self.legalLabels:
            self.calcPixelProbabilityGivenLabel(label)

        print("Finished calculating priors and likelihoods...")

    
    """
    Classifies a single datum
    """
    def predict(self, datum):
        posteriors = {}
        # Calculate the probability that the datum matches for each label
        for label in self.legalLabels:
            likelihood = self.likelihoods[label]
            pixelProbability = 0.0

            # Sums the probability of every single pixel
            for row in range(datum.height):
                for col in range(datum.width):
                    val = datum.getFeature(row, col)
                    prob = likelihood[row][col][val]
                    # Using log to prevent underflow error
                    pixelProbability += math.log(prob)
    
            posteriors[label] = self.priors[label] * pixelProbability

        # The posterior with the highest value is the prediction
        return max(posteriors, key=posteriors.get)


    """
    Takes in a list of datums and predicts a label for each one
    """
    def classify(self, testData):
        return [self.predict(datum) for datum in testData]
