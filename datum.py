import numpy as np


"""
The datum class holds the information of a single image
"""
class Datum:
  def __init__(self, data, width, height):
    self.width = width
    self.height = height
    self.pixels = data
    self.features = [[' ' for _ in range(width)] for _ in range(height)]
    self.extractFeatures(data)
    self.features = np.array(self.features)


  def getData(self):
    return self.pixels


  def getFeatures(self):
    return self.features


  def getFeature(self, row, col):
    return self.features[row][col]


  def extractFeatures(self, data):
    for rowIndex, row in enumerate(data):
      for charIndex in range(len(row)):
        self.features[rowIndex][charIndex] = self.convertChar(data[rowIndex][charIndex])


  def convertChar(self, char):
    if (char == '+') or (char == '#'):
      return 1
    else:
      return 0
