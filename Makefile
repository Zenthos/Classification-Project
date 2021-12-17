PYTHON=py -3

all:
	${PYTHON} perceptron.py
	${PYTHON} perceptron.py -d faces
	echo "---------------------------------------------------------------------------------------------"
	${PYTHON} naiveBayes.py
	${PYTHON} naiveBayes.py -d faces
	echo "---------------------------------------------------------------------------------------------"
	${PYTHON} mira.py
	${PYTHON} mira.py -d faces
	make clean

clean:
	del *.pyc
