import sys
import os
import scipy.io
import scipy.misc
from random import shuffle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import TanhLayer, SoftmaxLayer, LinearLayer, SigmoidLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection

# defining variables
n_features	= 784
n_layers	= 1
n_output	= 1
# network 	= buildNetwork(n_features, n_layers, n_output, bias=True, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
# dataset 	= SupervisedDataSet(n_features, n_output)
# trainer 	= BackpropTrainer(network, dataset)
n = FeedForwardNetwork()
inLayer = LinearLayer(n_features)
hiddenLayer = SigmoidLayer(n_layers)
outLayer = LinearLayer(1)
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)
n.sortModules()
dataset 	= SupervisedDataSet(n_features, n_output)

# loading dataset
mat 	= scipy.io.loadmat('../dataset/2012ME20780.mat')
data 	= mat["data_image" ]
labels 	= mat["data_labels"]
mat     = zip(data,labels)
shuffle(mat)
data	= zip(*mat)[0]
labels  = zip(*mat)[1]
	# seprating test and train data
train_d = data  [:1200]
train_l = labels[:1200]
test_d  = data  [1200:]
test_l  = labels[1200:]

# creating train dataset and training
for i,j in zip(train_d,train_l):
	dataset.addSample(i,(j,))
print "done"
trainer 	= BackpropTrainer(n, dataset)
trainer.train()
# trainer.trainUntilConvergence(verbose = True)

# predicting on test dataset
for i in test_d:
	print n.activate(tuple(i))