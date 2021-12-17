import numpy as np

class Datum:
  def __init__(self, data, width, height):
    self.width = width
    self.height = height
    self.pixels = data
    self.features = [[' ' for i in range(width)] for j in range(height)]
    self.extractFeatures(data)


  def getData(self):
    return self.pixels


  def getFeature(self, row, col):
    return self.features[row][col]


  def getFeatures(self):
    return np.array(self.features)


  def extractFeatures(self, data):
    for rowIndex, row in enumerate(data):
      for charIndex in range(len(row)):
        self.features[rowIndex][charIndex] = self.convertChar(data[rowIndex][charIndex])


  def convertChar(self, char):
    if (char == '+') or (char == '#'):
      return 1
    else:
      return 0
