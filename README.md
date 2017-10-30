# TensorflowConvolutionalNeuralNetworkModel

Author: Lars C. Schwensen
Date: 30/10/2017

# In short
This project provides a reusable convolutional neural network model written in python and Tensorflow.

# Environment:
Language: Python 3.6.2
OS: Ubuntu 16.04 LTS
Hardware: Intel I7-7700HQ, 16GiB DDR4, GeForce GTX 1050 Ti

# Dependencies:
Python3
Tensorflow
numpy
(cuda) ONLY for GPU support
(matplotlib) ONLY for the testing
(open-cv) ONLY for the testing

# Usage:
1. Import ConvolutionalNeuralNetworkModel
2. Instantiate a model, provide the inputData shape, dropout probability and the definition for each layer
3. Train the model by calling 'trainModel' with the trainingSet, trainingLabels, testingSet and testingLabels
4. Save the model, if you want.
5. Let the model predict by calling the function 'predict' 

# Additional Info:

source ~/tensorflow/bin/activate

ONLY for GPU support:
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
