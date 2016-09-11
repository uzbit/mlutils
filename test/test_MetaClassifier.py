
import unittest, os
import numpy
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from mlutils.MetaClassifier import MetaClassifier

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
		mc = MetaClassifier(self.params)
		mc.train(self.Xtrain, self.ytrain)
		
		preds = mc.predict(self.Xtest)
		print "Score: ", f1_score(self.ytest, preds, average='weighted')

if __name__ == '__main__':
	unittest.main()
