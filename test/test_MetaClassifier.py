
import unittest, os
import numpy as np
import dill as pickle
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from mlutils.MetaClassifier import MetaClassifier
from mlutils.mlutils import get_auc

SEED = 42

class MetaClassifierTest(unittest.TestCase):

	def setUp(self):
		self.testPath = os.path.dirname(os.path.realpath(__file__))
		iris = datasets.load_iris()
		self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
			iris.data, iris.target, test_size=0.1, stratify=iris.target, random_state=SEED
		)
		self.Xtrain = self.Xtrain.astype(np.float32)
		self.Xtest = self.Xtest.astype(np.float32)
		self.ytrain = self.ytrain.astype(np.int32)
		self.ytest = self.ytest.astype(np.int32)
		
		self.params = MetaClassifier.getDefaultParams()

	def test1(self):
		mcObj = MetaClassifier(self.params)
		mcObj.addKNC()
		mcObj.train(self.Xtrain, self.ytrain)
		
		print "AUC Score: ", get_auc(mcObj, self.Xtest, self.ytest)

	def test2(self):
		mcObj = MetaClassifier(self.params)
		mcObj.addKNC()
		print len(mcObj.getEstimatorList())
		
	def test3(self):
		mcObj = MetaClassifier(self.params)
		mcObj.addKNC()
		print len(mcObj.getEstimatorList())
		pickle.dump(mcObj, open('test.pickle', 'wb'))

if __name__ == '__main__':
	unittest.main()
