PYTHON=py -3

perceptron:
	${PYTHON} ./main.py -c perceptron
	${PYTHON} ./main.py -c perceptron -d faces

nb:
	${PYTHON} ./main.py -c nb
	${PYTHON} ./main.py -c nb -d faces

knn:
	${PYTHON} ./main.py -c knn
	${PYTHON} ./main.py -c knn -d faces
	make clean

clean:
	del *.pyc
