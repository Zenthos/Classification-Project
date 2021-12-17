import numpy as np

class Datum:
  def __init__(self, data, width, height):
    self.width = width
    self.height = height
    self.pixels = [[' ' for i in range(width)] for j in range(height)] 
    self.parseData(data)

  def getPixels(self):
    return np.array(self.pixels)

  def parseData(self, data):
    for rowIndex, row in enumerate(data):
      for charIndex in range(len(row)):
        self.pixels[rowIndex][charIndex] = self.convertChar(data[rowIndex][charIndex])

  def convertChar(self, char):
    if (char == '+') or (char == '#'):
      return 1
    else:
      return 0
