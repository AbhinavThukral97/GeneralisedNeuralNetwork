"""
Title: OCR Neural Network from scratch
Author: Abhinav Thukral
Language: Python
Dataset: MNIST
Description: Implementing a simpler, more readable version of OCR Neural Network from scratch to teach beginners. 
"""

import pandas as pd
import numpy as np
import time

#Reading the MNSIT dataset
def read_MNIST(path):
	data = pd.read_csv(path,skiprows=1)
	X = data.iloc[:, 1:]
	Y = data.iloc[:, 0]
	#Normalisation
	X = X.values/255
	Y = Y.values
	#10 values of MNIST dataset
	transformedY = [[0 for _ in range(10)] for x in range(len(Y))]
	#Transformation of Y
	for i in range(len(Y)):
		transformedY[i][Y[i]] = 1
	Y = transformedY
	"""Limiting data to run locally
	return X[0:4000], Y[0:4000]"""
	return X, Y

#Spliting data to test and train
def train_test_split(X,Y,split):
	l = len(X)
	limit = int(np.floor(split*l))
	#Since, MNSIT data is randomly arranged, we can simply slice
	x_train = X[0:limit]
	y_train = Y[0:limit]
	x_test = X[limit:l]
	y_test = Y[limit:l]
	return x_train,y_train,x_test,y_test

#ReLU activation function for hidden layers
def relu(input_layer):
	return np.maximum(0,input_layer)

#Softmax activation for output layers
def softmax(input_layer):
	exp_layer = np.exp(input_layer)
	softmax_layer = exp_layer/np.sum(exp_layer)
	return softmax_layer

#Using the structure of layers defined, initializing weight matrices
def generate_weights(layers):
	weights = []
	np.random.seed(1)
	for i in range(len(layers) - 1):
		#Adding 1 for bias
		w = 2*np.random.rand(layers[i]+1,layers[i+1]) - 1
		weights.append(w)
	return weights

#Feedforward network
def feedforward(x_vector,W):
	network = [np.append(1,np.array(x_vector))]
	for weight in W[:-1]:
		next_layer = relu(np.dot(network[-1],weight))
		network.append(np.append(1,next_layer))
	out_layer = softmax(np.dot(network[-1],W[-1]))
	network.append(out_layer)
	return network

#Backpropagation through the network
def backprop(network,y_vector,W,learning_rate):
	deltas = [np.subtract(network[-1],y_vector)]
	prev_layer = np.dot(W[-1],deltas[0])
	deltas.insert(0,prev_layer)
	for weight in list(reversed(W))[1:-1]:
		prev_layer = np.dot(weight,deltas[0][1:])
		deltas.insert(0,prev_layer)
	#Weight Update
	for l in range(len(W)):
		for i in range(len(W[l])):
			for j in range(len(W[l][i])):
				W[l][i][j] -= learning_rate*deltas[l][j]*network[l][i]

#Compute accuracy of the network for given weight parameters
def analyse_net(W,X,Y):
	correct_pred = 0
	for i in range(len(X)):
		y_pred = np.argmax(feedforward(X[i],W)[-1])
		if(y_pred==np.argmax(Y[i])):
			correct_pred+=1
	return np.round(correct_pred/i,4)

#Stochastic training for each training data
def train(x_train,y_train,W,epoch,learning_rate,x_test,y_test):
	for iteration in range(epoch):
		t0 = time.clock()
		for i in range(len(x_train)):
			network = feedforward(x_train[i],W)
			backprop(network,y_train[i],W,learning_rate)
		print("Epoch",iteration+1,"Accuracy",analyse_net(W,x_train,y_train),"Time",time.clock()-t0)

#Printing test data accuracy
def test_accuracy(x_test,y_test,W):
	print("Test Data Accuracy",analyse_net(W,x_test,y_test))

def run(hidden_layers,learning_rate,epoch,split):
	print("Epochs",epoch,"LR",learning_rate,"Hidden Layers",hidden_layers,"Split",split,sep="  ")
	X, Y = read_MNIST("MNISTtrain.csv")
	input_layer = len(X[0])
	output_layer = len(Y[0])
	layers = [input_layer] + hidden_layers + [output_layer]
	W = generate_weights(layers)
	x_train,y_train,x_test,y_test = train_test_split(X,Y,split)
	train(x_train,y_train,W,epoch,learning_rate,x_test,y_test)
	test_accuracy(x_test,y_test,W)

run([50],0.01,1,0.90)
"""
To implement a neural network with 2 hidden layers, with layer parameters 55, 25 respectively use:
run([55,25],0.003,30,0.9)
"""