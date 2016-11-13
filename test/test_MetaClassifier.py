
import unittest, os
import numpy as np
import cPickle as pickle
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
			iris.data, iris.target,
			test_size=0.2,
			stratify=iris.target,
			random_state=SEED
		)
		self.Xtrain = self.Xtrain.astype(np.float32)
		self.Xtest = self.Xtest.astype(np.float32)
		self.ytrain = self.ytrain.astype(np.int32)
		self.ytest = self.ytest.astype(np.int32)

	def test1(self):
		"""
			Test basic classifier addition, fit, prediction.
		"""
		mcObj = MetaClassifier()
		mcObj.addKNC(
			params={
				'n_jobs': -1
			}
		)
		mcObj.fit(self.Xtrain, self.ytrain)
		self.assertEqual(len(mcObj.getEstimatorList()), 1)
		self.assertEqual(round(get_auc(mcObj, self.Xtest, self.ytest), 2), 1.0)

	def test2(self):
		"""
			Test adding multiple classifiers, preprocessing functions,
			parameter passing (params is passed to the classifier __init__)
		"""
		mcObj = MetaClassifier(verbose=False)
		mcObj.addKNC(
			preproc='scale',
			params={'n_jobs': -1}
		)
		mcObj.addRFC(
			preproc=np.log,
			params={
				'n_estimators': 200,
			}
		)
		mcObj.fit(self.Xtrain, self.ytrain)
		self.assertEqual(round(get_auc(mcObj, self.Xtest, self.ytest), 2), 0.99)
		self.assertEqual(len(mcObj.getEstimatorList()), 2)

	def test3(self):
		"""
			Test save, load
		"""
		mcObj = MetaClassifier()
		mcObj.addKNC(
			params={
				'n_jobs': -1
			}
		)
		outFile = '/tmp/test.pickle'
		mcObj.fit(self.Xtrain, self.ytrain)

		# Save model
		pickle.dump(mcObj, open(outFile, 'wb'))
		self.assertTrue(os.path.exists(outFile))

		# Load model
		mcObj = pickle.load(open(outFile, 'rb'))
		self.assertEqual(round(get_auc(mcObj, self.Xtest, self.ytest), 2), 1.0)


if __name__ == '__main__':
	unittest.main()
