
import unittest, os
import numpy as np
import pickle as pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlutils.MetaClassifier import MetaClassifier
from mlutils.mlutils import get_auc

SEED = 42


class MetaClassifierTest(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.testPath = os.path.dirname(os.path.realpath(__file__))
		X, y = datasets.make_classification(n_samples=1000)
		cls.Xtrain, cls.Xtest, cls.ytrain, cls.ytest = train_test_split(
			X, y,
			test_size=0.3,
			stratify=y,
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
			params={'n_jobs': -1}
		)
		self.mcObj.fit(self.Xtrain, self.ytrain)
		self.assertEqual(len(self.mcObj.getEstimatorList()), 1)
		self.assertEqual(round(get_auc(self.mcObj, self.Xtest, self.ytest), 2), 0.89)

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
			preproc=np.abs, #lambda x: x**2,
			params={
				'n_estimators': 100,
				'n_jobs': -1,
			}
		)
		self.mcObj.setWeights([0.8, 0.2])

		self.mcObj.fit(self.Xtrain, self.ytrain)
		self.assertEqual(len(self.mcObj.getEstimatorList()), 2)
		self.assertEqual(round(get_auc(self.mcObj, self.Xtest, self.ytest), 2), 0.88)

	def test3(self):
		"""
			Test weighting of each classifier
		"""
		self.mcObj.addLR(
			params={}
		)
		self.mcObj.addLR(
			params={'C':10000}
		)
		self.mcObj.addKNC(
			params={'n_jobs': -1}
		)
		self.mcObj.addRFC(
			preproc=np.abs, #lambda x: x**2,
			params={
				'n_estimators': 100,
				'n_jobs': -1,
			}
		)
		numEst = len(self.mcObj.getEstimatorList())
		weights = np.ones(numEst)/float(numEst)

		# No weights, is equal weighting
		self.mcObj.setWeights([])
		self.mcObj.fit(self.Xtrain, self.ytrain)
		auc0 = get_auc(self.mcObj, self.Xtest, self.ytest)

		weights[0] *= 2.
		self.mcObj.setWeights(weights)
		self.mcObj.fit(self.Xtrain, self.ytrain)
		auc1 = get_auc(self.mcObj, self.Xtest, self.ytest)

		weights[0] /= 2.
		weights[-1] *= 2.
		self.mcObj.setWeights(weights)
		self.mcObj.fit(self.Xtrain, self.ytrain)
		auc2 = get_auc(self.mcObj, self.Xtest, self.ytest)

		#print auc0, auc1, auc2
		self.assertNotEqual(auc0, auc1)
		self.assertNotEqual(auc0, auc2)
		self.assertNotEqual(auc1, auc2)

	def test4(self):
		"""
			Test save, load
		"""
		self.mcObj.addKNC(
			preproc='scale',
			params={'n_jobs': -1}
		)
		self.mcObj.addETC(
			params={'n_jobs': -1}
		)
		self.mcObj.setWeights([0.8, 0.2])

		outFile = '/tmp/test.pickle'
		self.mcObj.fit(self.Xtrain, self.ytrain)
		auc1 = get_auc(self.mcObj, self.Xtest, self.ytest)

		# Save model
		pickle.dump(self.mcObj, open(outFile, 'wb'))
		self.assertTrue(os.path.exists(outFile))

		# Load model
		self.mcObj = pickle.load(open(outFile, 'rb'))
		auc2 = get_auc(self.mcObj, self.Xtest, self.ytest)

		# Same result before and after saving/loading
		self.assertEqual(auc1, auc2)


if __name__ == '__main__':
	unittest.main()
