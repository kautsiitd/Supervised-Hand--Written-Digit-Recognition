import sys
import os
import scipy.io
import scipy.misc
from sklearn import datasets
from numpy import ravel
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

mat 	= scipy.io.loadmat('../dataset/2012ME20780.mat')
data 	= mat["data_image" ]
labels 	= mat["data_labels"]
# olivetti = datasets.fetch_olivetti_faces()
X, y = data, labels
print X.shape

ds = ClassificationDataSet(784, 1, nb_classes = 10)
for k in xrange(len(X)):
    ds.addSample(ravel(X[k]), y[k])

tstdata, trndata = ds.splitWithProportion(0.25)
# trndata._convertToOneOfMany()
# tstdata._convertToOneOfMany()
tstdata = ClassificationDataSet(2, 1, nb_classes=10)
for n in xrange(0, tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(2, 1, nb_classes=10)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
print trndata['input'], trndata['target'], tstdata.indim, tstdata.outdim

fnn = buildNetwork(trndata.indim, 28, trndata.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)

if os.path.isfile('oliv.xml'):
	fnn = NetworkReader.readFrom('oliv.xml')
else:
	fnn = buildNetwork(trndata.indim, 28, trndata.outdim, outclass=SoftmaxLayer)

NetworkWriter.writeToFile(fnn, 'oliv.xml')
trainer.trainEpochs(50)
print 'Percent Error on Test dataset: ',percentError(trainer.testOnClassData(
			dataset=tstdata)
			, tstdata['class'] )
