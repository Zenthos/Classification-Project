# Machine-Learning
Final Project for CS440, Introduction to Artificial Intelligience, inspired by the [UC Berkeley Project](http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/classification.html).

Implementations of the following algorithms
- Single Layer Perceptron
- Multinomial Naive Bayes
- K-Nearest Neighbor

Each model can recognize digits and faces at about a 65% - 80% accuracy.

## Commands

| Algorithm          | Command                         |
| ------------------ | ------------------------------- |
| Perceptron         | python ./main.py -c perceptron  |
| Naive Bayes        | python ./main.py -c nb          |
| K-Nearest Neighbor | python ./main.py -c knn         |

To make the algorithm learn to recognize faces, append to the end of the command
> -d faces
