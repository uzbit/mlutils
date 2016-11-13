
import unittest, os
import numpy as np
import cPickle as pickle
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from mlutils.MetaClassifier import MetaClassifier
from mlutils.mlutils import get_auc

SEED = 42

class MetaClassifierTest(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.testPath = os.path.dirname(os.path.realpath(__file__))
		iris = datasets.load_iris()
		cls.Xtrain, cls.Xtest, cls.ytrain, cls.ytest = train_test_split(
			iris.data, iris.target,
			test_size=0.2,
			stratify=iris.target,
			random_state=SEED
		)
		cls.Xtrain = cls.Xtrain.astype(np.float32)
		cls.Xtest = cls.Xtest.astype(np.float32)
		cls.ytrain = cls.ytrain.astype(np.int32)
		cls.ytest = cls.ytest.astype(np.int32)

	def setUp(self):
		self.mcObj = MetaClassifier(verbose=False)

	def test1(self):
		"""
			Test basic classifier addition, fit, prediction.
		"""
		self.mcObj.addKNC(
			params={
				'n_jobs': -1
			}
		)
		self.mcObj.fit(self.Xtrain, self.ytrain)
		self.assertEqual(len(self.mcObj.getEstimatorList()), 1)
		self.assertEqual(round(get_auc(self.mcObj, self.Xtest, self.ytest), 2), 1.0)

	def test2(self):
		"""
			Test adding multiple classifiers, preprocessing functions,
			parameter passing (params is passed to the classifier __init__).
			preproc function is applied to each feature vector before training
			and prediction. preproc='scale' is sklearn StandardScaler.
		"""
		self.mcObj.addKNC(
			preproc='scale',
			params={'n_jobs': -1}
		)
		self.mcObj.addRFC(
			preproc=np.log,
			params={
				'n_estimators': 200,
			}
		)
		self.mcObj.fit(self.Xtrain, self.ytrain)
		self.assertEqual(round(get_auc(self.mcObj, self.Xtest, self.ytest), 2), 0.99)
		self.assertEqual(len(self.mcObj.getEstimatorList()), 2)

	def test3(self):
		"""
			Test save, load
		"""
		self.mcObj.addKNC(
			preproc='scale',
			params={
				'n_jobs': -1
			}
		)
		outFile = '/tmp/test.pickle'
		self.mcObj.fit(self.Xtrain, self.ytrain)

		# Save model
		pickle.dump(self.mcObj, open(outFile, 'wb'))
		self.assertTrue(os.path.exists(outFile))

		# Load model
		self.mcObj = pickle.load(open(outFile, 'rb'))
		self.assertEqual(round(get_auc(self.mcObj, self.Xtest, self.ytest), 2), 0.99)


if __name__ == '__main__':
	unittest.main()
