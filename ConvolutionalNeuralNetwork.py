#####################################
#									#
#	Author: Lars Schwensen			#
#	Date: 30.10.2017				#
#									#
#	Module for defining a 			#
#	convolutional neural network 	#
#									#
#####################################

import tensorflow as tf
import numpy as np
from enum import Enum

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Type(Enum):
	CONV = 0
	FULLY = 1
	OUTPUT = 2

#Helperclass for defining every layer
class Layer:
	def __init__(self,layerType,inputSize,outputSize,convFilter={'x':5,'y':5},convStrides=[1,1,1,1],poolStrides=[1,2,2,1],poolKSize=[1,2,2,1],convPadding='SAME',poolPadding='SAME'):
		self.layerType = layerType
		self.inputSize = inputSize
		self.outputSize = outputSize

		self.convFilter = convFilter

		self.convStrides = convStrides
		self.poolStrides = poolStrides

		self.poolKSize = poolKSize

		self.convPadding = convPadding
		self.poolPadding = poolPadding

#Definition of a convolutional layer
class ConvolutionalLayer:
	def __init__(self,previous,layerDefinition):
		self.weights = tf.Variable(tf.random_normal([layerDefinition.convFilter['x'],layerDefinition.convFilter['y'],layerDefinition.inputSize,layerDefinition.outputSize]))
		self.biases = tf.Variable(tf.random_normal([layerDefinition.outputSize]))

		self.layer = tf.nn.conv2d(previous,self.weights, strides=layerDefinition.convStrides,padding=layerDefinition.convPadding)
		self.layer = tf.nn.bias_add(self.layer, self.biases)
		self.layer = tf.nn.relu(self.layer)

		self.layer = tf.nn.max_pool(self.layer, ksize=layerDefinition.poolKSize, strides=layerDefinition.poolStrides, padding=layerDefinition.poolPadding)

#Definition of a fully connected layer		
class FullyConnectedLayer:
	def __init__(self,previous,keepProbability,layerDefinition):
		self.weights = tf.Variable(tf.random_normal([layerDefinition.inputSize,layerDefinition.outputSize]))
		self.biases = tf.Variable(tf.random_normal([layerDefinition.outputSize]))

		self.layer = tf.reshape(previous,[-1,layerDefinition.inputSize])
		self.layer = tf.nn.relu(tf.matmul(self.layer,self.weights) + self.biases)

		self.layer = tf.nn.dropout(self.layer, keepProbability)

#Definition of an output layer. The last layer of the network must be an output layer!!!
class OutputLayer:
	def __init__(self,previous,layerDefinition):
		self.weights = tf.Variable(tf.random_normal([layerDefinition.inputSize,layerDefinition.outputSize]))
		self.biases = tf.Variable(tf.random_normal([layerDefinition.outputSize]))

		self.layer = tf.matmul(previous,self.weights) + self.biases

#an instance of this class creates a new model.
#inputDataShape: defines the 2D-shape of the date to process
#dropoutProb: defines the dropout rate
#layers: contains the definition for each layer of the net
#sess: allows to pass a global tensorflow session
class ConvolutionalNeuralNetworkModel:
	def __init__(self,inputDataShape=[28,28],dropoutProb=0.5,layers=[],sess=None):
		if len(layers) == 0:
			raise ValueError('NO LAYERS DEFINED!!!')
		elif layers[-1].layerType != Type.OUTPUT:
			raise ValueError('THE LAST LAYER MUST BE AN OUTPUTLAYER!!!')
		elif len(inputDataShape) != 2:
			raise ValueError('PLEASE DEFINE THE inputDataShape as [x,y]!!!')

		self.inputDataShape = inputDataShape
		self.dropoutProb = dropoutProb

		self.inputData = tf.placeholder('float',[None, abs(np.prod(inputDataShape))])
		self.inputLabel = tf.placeholder('float',[None, layers[-1].outputSize])
		self.keepProbability = tf.placeholder('float') 

		self.sess = sess
		if self.sess == None:
			self.sess = tf.Session(config=config)

		self.layer = []
	
		#define the actual model
		self.model = self.defineModel(layers)
	
	def defineModel(self, layers):
		self.inputData = tf.reshape(self.inputData,[-1,self.inputDataShape[0],self.inputDataShape[1],1])	

		for layerDefinition in layers:
			previous = self.inputData
			if len(self.layer) > 0:
				previous = self.layer[-1]

			if layerDefinition.layerType == Type.CONV: #define a convolutional layer
				self.layer.append(ConvolutionalLayer(previous,layerDefinition).layer)
			elif layerDefinition.layerType == Type.FULLY: #define a fully connected layer
				self.layer.append(FullyConnectedLayer(previous,self.keepProbability,layerDefinition).layer)
			elif layerDefinition.layerType == Type.OUTPUT: # define an output layer
				if(layerDefinition != layers[-1]):
					raise ValueError('THE OUTPUTLAYER MUST BE DEFINED AT LAST!!!')
				else:
					self.layer.append(OutputLayer(previous,layerDefinition).layer)
			else:
				raise ValueError('LAYER DEFINITION INCONSISTENT!!!')

		#return the outputlayer
		return self.layer[-1]

	#Train the model 
	#feedForwardCycles: defines the number of cycles for feedForward and Backpropagation
	#batchsize: defines the number of training elements per time. This strongly depends on RAM resources of GPU
	#debugInfo: allows to provide some information during the training
	def trainModel(self,trainingSet,trainingLabels, testingSet, testingLabels, feedForwardCycles=10, batchSize=100,debugInfo=False):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model,labels=self.inputLabel))
		optimizer = tf.train.AdamOptimizer().minimize(cost)
		
		correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.inputLabel, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred,'float'))

		#initialize the variables
		self.sess.run(tf.global_variables_initializer())

		#Cycles of feed forward + backpropagation
		for epoch in range(feedForwardCycles):
			if(debugInfo):
				print('Cycle:', (epoch+1), '/', feedForwardCycles)
			
			i = 0
			while i < len(trainingSet):	
				start = i
				end = i+batchSize
				if end > len(trainingSet):
					end = len(trainingSet)

				batchTrainingSet = np.array(trainingSet[start:end]) 	
				batchTrainingSet = np.reshape(batchTrainingSet,[-1,self.inputDataShape[0],self.inputDataShape[1],1])

				batchTrainingLabels = np.array(trainingLabels[start:end]) 

				self.sess.run([optimizer], feed_dict={self.inputData:batchTrainingSet, self.inputLabel:batchTrainingLabels,self.keepProbability:self.dropoutProb})
				
				if(debugInfo):
					if i % (batchSize *10) == 0 or end == len(trainingSet):
						loss, acc = self.sess.run([cost,accuracy],feed_dict={self.inputData:batchTrainingSet, self.inputLabel:batchTrainingLabels,self.keepProbability:1.})
						print('input', i, 'loss:', loss, 'acc:', acc)

				i += batchSize


		#define the condition: The condition is true when the label has a 1 
		#on the same index where the model predicts the highest probability
		#argmax returns the index of the highest value of the array
		condition = tf.equal(tf.argmax(self.model,1), tf.argmax(self.inputLabel,1))
		
		#define the format of the accuracy
		accuracyFormat = tf.reduce_mean(tf.cast(condition,'float'))

		#calculate the accuracy
		with self.sess.as_default():	
			testingSet = np.reshape(testingSet,[-1,self.inputDataShape[0],self.inputDataShape[1],1])		
			acc = accuracyFormat.eval({self.inputData:testingSet, self.inputLabel:testingLabels,self.keepProbability:1.})
					
			print('Accuracy', acc)

		return accuracy

	def saveModel(self, name):
		saver = tf.train.Saver()
		saver.save(self.sess, name)

	def loadModel(self, name):
		saver = tf.train.Saver()
		saver.restore(self.sess, name)

	def close(self):
		self.sess.close()

	def predict(self,data):	
		prediction = self.sess.run(self.model,feed_dict={self.inputData:data,self.keepProbability:1.})
		sum = 0
		for i in range(len(prediction[0])):
			if prediction[0][i] >= 0:
				sum += prediction[0][i]
			else:
				prediction[0][i] = 0

		if sum != 0:
			prediction[0] = prediction[0] / sum
			
		return prediction, np.argmax(prediction[0])

#layerDefinition = []
#layerDefinition.append(Layer(Type.CONV, 1, 32))
#layerDefinition.append(Layer(Type.CONV, 32, 64))
#layerDefinition.append(Layer(Type.FULLY, 7*7*64, 1024))
#layerDefinition.append(Layer(Type.OUTPUT, 1024, 10))

#myModel = NeuralNetworkModel(layers=layerDefinition)