from ConvolutionalNeuralNetwork import NeuralNetworkModel
from ConvolutionalNeuralNetwork import Layer
from ConvolutionalNeuralNetwork import Type
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def generateImage(file):		
	img = Image.open(file).convert('LA')
	img.save(file)

	x = cv.imread(file,0)
	data = np.asarray((255-x)/255.0)
	return data, np.reshape(data, (-1,28,28,1))


mnist = input_data.read_data_sets("data",one_hot=True)

trainingSet = mnist.train.images 
trainingLabels = mnist.train.labels

testingSet = mnist.test.images 
testingLabels = mnist.test.labels

layerDefinition = []
layerDefinition.append(Layer(Type.CONV, 1, 32))
layerDefinition.append(Layer(Type.CONV, 32, 64))
layerDefinition.append(Layer(Type.FULLY, 7*7*64, 1024))
layerDefinition.append(Layer(Type.OUTPUT, 1024, 10))

myModel = NeuralNetworkModel(layers=layerDefinition)

#Select this if you want to train a new model
myModel.trainModel(trainingSet,trainingLabels, testingSet, testingLabels, feedForwardCycles=5, batchSize=100,debugInfo=False)
myModel.saveModel('mnistModel')

#Select this if you already trained a model which you want to reuse
#myModel.loadModel('mnistModel')

images = ['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']

for num in images:
	number,numberArray = generateImage('Examples/%s.png' % (num))
	prediction, highestProbability = myModel.predict(numberArray)
	plt.title('Your digit is probably a %d' % (highestProbability))
	plt.imshow(number)
	plt.show()
