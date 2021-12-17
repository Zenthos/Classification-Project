import dataParser as dp
from operator import itemgetter


"""
K Nearest Neighbor Classifier.
"""
class KNearestNeighborClassifier:
    def __init__(self, legalLabels, maxIterations):
        self.legalLabels = legalLabels
        self.type = "K-Nearest Neighbor"
        self.k = 100


    """
    No actual training occurs, the algorithm works during classification
    """
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels


    """
    Calculates the test image's distance from the trainingImage
    """
    def distance(self, testImage, trainingImage): 
        total_distance = 0
        
        # Increase the image's distance for every pixel that doesn't match
        for row in range(testImage.height):
            for char in range(testImage.width):
                if testImage.getFeature(row, char) != trainingImage.getFeature(row, char):
                    total_distance += 1

        return total_distance


    """
    Classifies each datum and attempts to predict which label matches the most
    """
    def predict(self, datum):
        # Calculate the distances of each trainingImage with the datum we wanna predict
        distances = []
        for index, trainingImage in enumerate(self.trainingData):
            distances.append((self.trainingLabels[index], self.distance(datum, trainingImage)))

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


# Training with 5000 digits
if __name__ == "__main__": 
    # Parses the datafiles into a list of Datum Objects, each holding a np.2d array size 28x28
    trainingData = dp.loadDataFile("digitdata/trainingimages", 5000, 28, 28)
    trainingLabels = dp.loadLabelFile("digitdata/traininglabels", 5000)

    # Needs to classify handwritten digits 0-9
    test = KNearestNeighborClassifier(range(10), 0)
    test.train(trainingData, trainingLabels, [], [])