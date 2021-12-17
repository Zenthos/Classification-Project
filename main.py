import optparse
import samples
import util
import sys
import time

from perceptron import PerceptronClassifier
from naiveBayes import NaiveBayesClassifier

"""
Global Variables
"""
DATUM_WIDTH = 28
DATUM_HEIGHT = 28


"""
Returns a set of pixel features indicating whether
each pixel in the provided datum is white (0) or gray/black (1)
"""
def digitFeatureExtractor(datum):
    features = util.Counter()
    for x in range(DATUM_WIDTH):
        for y in range(DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0

    return features


"""
Returns a set of pixel features indicating whether
each pixel in the provided datum is an edge (1) or no edge (0)
"""
def faceFeatureExtractor(datum):
    features = util.Counter()
    for x in range(DATUM_WIDTH):
        for y in range(DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0

    return features


"""
Main Function
"""
if __name__ == "__main__":
    parser = optparse.OptionParser() 
    parser.add_option('-d', '--data', choices=['digits', 'faces'], default='digits')
    parser.add_option('-c', '--classifier', choices=['perceptron', 'nb'], default='perceptron')
    options = parser.parse_args(sys.argv[1:])[0]
    maxIterations = 2

    if (options.data == 'faces'):
      legalLabels = range(2)
    else:
      legalLabels = range(10)

    if (options.classifier == 'nb'):
      classifier = NaiveBayesClassifier(legalLabels, maxIterations)
    else:
      classifier = PerceptronClassifier(legalLabels, maxIterations)

    print "\nStarting", classifier.type, "Classifier. You have chosen to classify", options.data,"!\n"
    if (options.data == "faces"):
        DATUM_WIDTH = 60
        DATUM_HEIGHT = 70

    rawTrainingData = []
    trainingLabels = []
    rawValidationData = []
    validationLabels = []
    rawTestData = []
    testLabels = []

    print "Loading,", options.data,"data..."
    if (options.data == "digits"):
        trainingCount = 5000
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", trainingCount, DATUM_WIDTH, DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", trainingCount)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", 1000, DATUM_WIDTH, DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", 1000)
        rawTestData = samples.loadDataFile("digitdata/testimages", 1000, DATUM_WIDTH, DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", 1000)
    else:
        trainingCount = 450
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", trainingCount, DATUM_WIDTH, DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", trainingCount)
        rawValidationData = samples.loadDataFile("facedata/facedatavalidation", 300, DATUM_WIDTH, DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 300)
        rawTestData = samples.loadDataFile("facedata/facedatatest", 150, DATUM_WIDTH, DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", 150)

    print "Parsing,", options.data,"data..."
    featureExtractor = digitFeatureExtractor;
    if (options.data == "faces"):
        featureExtractor = faceFeatureExtractor;

    trainingData = map(featureExtractor, rawTrainingData)
    validationData = map(featureExtractor, rawValidationData)
    testData = map(featureExtractor, rawTestData)

    currentPercentage = 0
    percentages = []
    trainingTimes = []
    validationAccuracies = []
    testingAccuracies = []

    for k, i in enumerate(xrange(0, len(trainingData), (len(trainingData) / 10))):      
      trainingChunk = trainingData[i:i + (len(trainingData) / 10)]
      trainingLabelsChunk = trainingLabels[i:i + (len(trainingLabels) / 10)]

      valChunkLength = (len(validationData) / 10)
      validationChunk = validationData[(k * valChunkLength):(k * valChunkLength) + (len(validationData) / 10)]
      validationLabelsChunk = validationLabels[(k * valChunkLength):(k * valChunkLength) + (len(validationLabels) / 10)]

      currentPercentage += 10
      percentages.append(currentPercentage)

      print "\nTraining '%s' classifier to recognize '%s' from %d%% of data..." % (classifier.type, options.data, currentPercentage)
      trainingStart = time.time()
      classifier.train(trainingChunk, trainingLabelsChunk, validationChunk, validationLabelsChunk)
      trainingTime = time.time() - trainingStart
      trainingTimes.append(trainingTime)

      print "Calculating validation accuracy..."
      validatingStart = time.time()
      guesses = classifier.classify(validationData)
      correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
      validationAccuracy = (100.0 * correct / len(testLabels))
      # print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % validationAccuracy
      validatingTime = time.time() - validatingStart
      validationAccuracies.append(validationAccuracy)

      print "Calculating testing accuracy..."
      testingStart = time.time()
      guesses = classifier.classify(testData)
      correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
      testAccuracy = (100.0 * correct / len(testLabels))
      # print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % testAccuracy
      testingTime = time.time() - testingStart
      testingAccuracies.append(testAccuracy)

    # Tabulate Data
    print "\n{:<10} {:<20} {:<20} {:<20}".format('Percent', 'Training Time', 'Validation Acc', 'Testing Acc')
    print "----------------------------------------------------------------"
    for i in range(10):
      percent = "%s%%" % (percentages[i])
      t_time = "%.2fs" % (trainingTimes[i])
      v_acc = "%.2f%%" % (validationAccuracies[i])
      t_acc = "%.2f%%" % (testingAccuracies[i])
      print "{:<10} {:<20} {:<20} {:<20}".format(percent, t_time, v_acc, t_acc)

    print "----------------------------------------------------------------\n"