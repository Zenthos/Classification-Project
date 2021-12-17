import dataParser as dp
import optparse
import time
import sys

from nearest import KNearestNeighborClassifier
from perceptron import PerceptronClassifier
from naiveBayes import NaiveBayesClassifier


"""
Main Function
"""
if __name__ == "__main__":
    parser = optparse.OptionParser() 
    parser.add_option('-d', '--data', choices=['digits', 'faces'], default='digits')
    parser.add_option('-c', '--classifier', choices=['nb', 'perceptron', 'knn'], default='nb')
    options = parser.parse_args(sys.argv[1:])[0]
    maxIterations = 2

    if (options.data == 'faces'):
      legalLabels = range(2)
    else:
      legalLabels = range(10)

    if (options.classifier == 'nb'):
      classifier = NaiveBayesClassifier(legalLabels, maxIterations)
    elif (options.classifier == 'knn'):
      classifier = KNearestNeighborClassifier(legalLabels, maxIterations)
    else:
      classifier = PerceptronClassifier(legalLabels, maxIterations)

    print("\nStarting", classifier.type, "Classifier. You have chosen to classify", options.data,"!\n")

    DATUM_WIDTH = 28
    DATUM_HEIGHT = 28
    if (options.data == "faces"):
        DATUM_WIDTH = 60
        DATUM_HEIGHT = 70

    trainingData = []
    trainingLabels = []
    validationData = []
    validationLabels = []
    testData = []
    testLabels = []

    print("Loading and parsing,", options.data,"data...")
    if (options.data == "digits"):
        trainingCount = 5000
        trainingData = dp.loadDataFile("digitdata/trainingimages", trainingCount, DATUM_WIDTH, DATUM_HEIGHT)
        trainingLabels = dp.loadLabelFile("digitdata/traininglabels", trainingCount)
        validationData = dp.loadDataFile("digitdata/validationimages", 1000, DATUM_WIDTH, DATUM_HEIGHT)
        validationLabels = dp.loadLabelFile("digitdata/validationlabels", 1000)
        testData = dp.loadDataFile("digitdata/testimages", 1000, DATUM_WIDTH, DATUM_HEIGHT)
        testLabels = dp.loadLabelFile("digitdata/testlabels", 1000)
    else:
        trainingCount = 450
        trainingData = dp.loadDataFile("facedata/facedatatrain", trainingCount, DATUM_WIDTH, DATUM_HEIGHT)
        trainingLabels = dp.loadLabelFile("facedata/facedatatrainlabels", trainingCount)
        validationData = dp.loadDataFile("facedata/facedatavalidation", 300, DATUM_WIDTH, DATUM_HEIGHT)
        validationLabels = dp.loadLabelFile("facedata/facedatatrainlabels", 300)
        testData = dp.loadDataFile("facedata/facedatatest", 150, DATUM_WIDTH, DATUM_HEIGHT)
        testLabels = dp.loadLabelFile("facedata/facedatatestlabels", 150)

    currentPercentage = 0
    percentages = []
    trainingTimes = []
    validationAccuracies = []
    validationTimes = []
    testingAccuracies = []
    testingTimes = []

    # This is for testing the classifiers performance at 10% of the data 20%, 30%, etc...
    for k, i in enumerate(range(0, len(trainingData), int(len(trainingData) / 10))):      
      trainingChunk = trainingData[i:i + int(len(trainingData) / 10)]
      trainingLabelsChunk = trainingLabels[i:i + int(len(trainingLabels) / 10)]

      valChunkLength = int(len(validationData) / 10)
      validationChunk = validationData[(k * valChunkLength):(k * valChunkLength) + int(len(validationData) / 10)]
      validationLabelsChunk = validationLabels[(k * valChunkLength):(k * valChunkLength) + int(len(validationLabels) / 10)]

      currentPercentage += 10
      percentages.append(currentPercentage)

      print("\nTraining '%s' classifier to recognize '%s' from %d%% of data..." % (classifier.type, options.data, currentPercentage))
      trainingStart = time.time()
      
      if classifier.type == 'nb':
        classifier.train(trainingData[0:(k * valChunkLength)], trainingLabels[0:(k * valChunkLength)])
      else:
        classifier.train(trainingChunk, trainingLabelsChunk, validationChunk, validationLabelsChunk)
      
      trainingTime = time.time() - trainingStart
      trainingTimes.append(trainingTime)

      print("Calculating validation accuracy...")
      validatingStart = time.time()
      guesses = classifier.classify(validationData)
      correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
      validationAccuracy = (100.0 * correct / len(testLabels))
      print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % validationAccuracy)
      validationTimes.append(time.time() - validatingStart)
      validationAccuracies.append(validationAccuracy)

      print("Calculating testing accuracy...")
      testingStart = time.time()
      guesses = classifier.classify(testData)
      correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
      testAccuracy = (100.0 * correct / len(testLabels))
      print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % testAccuracy)
      testingTimes.append(time.time() - testingStart)
      testingAccuracies.append(testAccuracy)

    # Tabulate Data
    print("\n{:<10} {:<20} {:<20} {:<20} {:<20} {:<20}".format(
      'Percent', 
      'Training Time', 
      'Validation Acc', 
      'Validation Times', 
      'Testing Acc', 
      'Testing Times'
    ))

    print("------------------------------------------------------------------------------------------------------------")
    for i in range(10):
      percent = "%s%%" % (percentages[i])
      t_time = "%.2fs" % (trainingTimes[i])
      v_acc = "%.2f%%" % (validationAccuracies[i])
      v_time = "%.2fs" % (validationTimes[i])
      t_acc = "%.2f%%" % (testingAccuracies[i])
      test_time = "%.2fs" % (testingTimes[i])
      print("{:<10} {:<20} {:<20} {:<20} {:<20} {:<20}".format(percent, t_time, v_acc, t_acc))

    print("------------------------------------------------------------------------------------------------------------\n")
