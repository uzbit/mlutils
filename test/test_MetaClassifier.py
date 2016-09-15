
import unittest, os
import numpy
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from mlutils.MetaClassifier import MetaClassifier
from mlutils.mlutils import get_auc

class MetaClassifierTest(unittest.TestCase):

	def setUp(self):
		self.testPath = os.path.dirname(os.path.realpath(__file__))
		iris = datasets.load_iris()
		self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
			iris.data, iris.target, test_size=0.1, stratify=iris.target, random_state=42
		)
		self.Xtrain = self.Xtrain.astype(numpy.float32)
		self.Xtest = self.Xtest.astype(numpy.float32)
		self.ytrain = self.ytrain.astype(numpy.int32)
		self.ytest = self.ytest.astype(numpy.int32)
		
		self.params = MetaClassifier.getDefaultParams()

	def test1(self):
		mcObj = MetaClassifier(self.params)
		mcObj.addKNC()
		mcObj.train(self.Xtrain, self.ytrain)
		
		print "AUC Score: ", get_auc(mcObj, self.Xtest, self.ytest)

if __name__ == '__main__':
	unittest.main()
