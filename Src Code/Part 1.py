from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer,TanhLayer
import os
import csv

def load_dataset(dataset, X, y):
    enc = OneHotEncoder(n_values=10)
    yenc = enc.fit_transform(np.matrix(y)).todense()
    for i in range(y.shape[0]):
        dataset.addSample(X[i, :], yenc[i][0])

LEARNING_RATE = .001
NUM_EPOCHS = 50
NUM_HIDDEN_UNITS = 25

print "Loading MATLAB data..."    
data = scipy.io.loadmat("../dataset/2012ME20780.mat")
X = data["data_image"]
y = data["data_labels"]
n_features = X.shape[1]
n_classes = len(np.unique(y))

# split up training data for cross validation
print "Split data into training and test sets..."
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, 
                                                random_state=42)
ds_train = ClassificationDataSet(X.shape[1], 10)
load_dataset(ds_train, Xtrain, ytrain)

# build a 400 x 25 x 10 Neural Network
print "Building %d x %d x %d neural network..." % (n_features, NUM_HIDDEN_UNITS, n_classes)
fnn = buildNetwork(n_features, NUM_HIDDEN_UNITS, n_classes, bias=True, 
                   hiddenclass=TanhLayer, outclass=SoftmaxLayer)
print fnn

# train network
print "Training network..."
trainer = BackpropTrainer(fnn, ds_train, learningrate=LEARNING_RATE)
for i in range(NUM_EPOCHS):
    error = trainer.train()
    print "Epoch: %d, Error: %7.4f" % (i, error)
# error = trainer.trainUntilConvergence()

# predict using test data
print "Making predictions..."
ypreds = []
ytrues = []
for i in range(Xtest.shape[0]):
    pred = fnn.activate(Xtest[i, :])
    ypreds.append(pred.argmax())
    ytrues.append(ytest[i][0])
print "Accuracy on test set: %7.4f" % accuracy_score(ytrues, ypreds, normalize=True)

c_label	  = [0]*10
frqs	= [['Actual/Predict',0,1,2,3,4,5,6,7,8,9,'Recall']]+[[i]+[0 for j in range(10)] for i in range(10)]+[['Precision']]
# saving results
for i in range(Xtest.shape[0]):
	dir_name = "../Output/3.b/"+str(ypreds[i])
	try:
		os.mkdir(dir_name)
	except:pass
	scipy.misc.imsave(dir_name+"/"+str(c_label[ypreds[i]])+".bmp",Xtest[i].reshape(28,28))
	c_label[ypreds[i]]	+= 1
	frqs[ypreds[i]+1][ytest[i]+1] += 1

for i in range(1,11):
	if sum(frqs[i]) != 0:
		frqs[ i].append(round((frqs[i][i]*100.0)/sum(frqs[i]),2))
	else:
		frqs[ i].append('inf')
	if sum(zip(*frqs[1:-1])[i]) != 0:
		frqs[-1].append(round((frqs[i][i]*100.0)/sum(zip(*frqs[1:-1])[i]),2))
	else:
		frqs[-1].append('inf')

with open("../Output/3.b/Result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(frqs)