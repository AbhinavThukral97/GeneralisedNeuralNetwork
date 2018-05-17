# OCR Neural Network from scratch
Generalised neural network implemented from scratch in python to teach step-wise functioning of a neural network and backpropagation training algorithm. 

## Understanding Neural Networks

Simply put, Artificial Neural Network (ANN) is a network of nodes and interconnections that can be used to solve machine learning problems.
Basic terminology: 
- **Layer**
  The structure of the network is split into layers, the first being the input vector and the last representing the output vector/value.
- **Neuron/Node**
  The basic building block of the network. This receives inputs from nodes in the previous layer and feeds its output to the nodes in the next layer
- **Weights**
  Each interconnection between the neurons has an associated weight. These weights are the parameters responsible for learning. 
  Basically, these weights represent the strength of each connection. Why do we need this? Because these weights help transform the input to the final output. If we find the right values for these weights, we can generate a desired output from the given inputs.
- **Activation Function**
This is simple function that is applied at each neuron that determines the output of the neuron according to its input. Say, the neuron is supposed to output '1' only when the input is positive, otherwise it should output '0'. Why do we need them now? Because they can introduce non-linearities between nodes that can help us find solutions to all kinds of problems.

## Backpropagation Intuition (Why does this work)

The basic intuition behind backpropagation is that we know our output is a function of the inputs and the weights in the network. We know the input and we know the desired output. We want to change our weights such that our inputs are transformed to our desired output with the least error. 
So what do we do, hmm... 
Well, we know our objective. We want to minimize this error between our generated output and our target output. 

## About the OCR implementation


## Using the generalised code

The number of layers and the size of each layer i.e. the number of nodes in each layer are to be specified as an array (list).

Eg. A neural network with _input layer -> hidden layer of size 100 -> hidden layer of size 25 -> output layer_, will be represented as `[100,25]`.  Implicitly, the number of hidden layers will be the length of this list.

The hyperparameters are network structure, learning rate, epochs, train-test data split. Change the source of your dataset and simply run after setting these parameters.
