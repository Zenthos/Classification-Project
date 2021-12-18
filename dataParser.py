from datum import Datum
import numpy as np


"""
Read all the images in the file, and return list of n Datum Objects
"""
def loadDataFile(fileName, n, width, height):
    items = []
    with open(fileName, 'r') as infile:
        lines = infile.readlines()[::-1]
        for i in range(n):
            data = []

            for _ in range(height):
                data.append(list(lines.pop().replace('\n', '')))

            if len(data[0]) < width - 1:
                print("Truncating at %d examples (maximum)" % i)
                break

            items.append(Datum(data, width, height))

    return np.array(items)


"""
Read all the lines in the file, and return a list of n items
"""
def loadLabelFile(fileName, n):
    labels = []
    with open(fileName, 'r') as infile:
        lines = infile.readlines()

        for i in range(n):
            line = lines[i].replace('\n', '')
            if line == '':
                break
            labels.append(int(line))

    return np.array(labels)
