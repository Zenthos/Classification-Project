from datum import Datum

def loadDataFile(fileName, n, width, height):
  items = []
  with open(fileName, 'r') as infile:
    lines = infile.readlines()
    for i in range(n):
      data = []

      for _ in range(height):
        data.append(list(lines.pop().replace('\n', '')))

      if len(data[0]) < width - 1:
        print("Truncating at %d examples (maximum)" % i)
        break

      items.append(Datum(data, width, height))

  return items

def loadLabelFile(fileName, n):
  labels = []
  with open(fileName, 'r') as infile:
    lines = infile.readlines()

    for i in range(n):
      line = lines[i].replace('\n', '')
      if line == '': 
        break
      labels.append(int(line))

  return labels