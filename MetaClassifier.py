import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
	from treeinterpreter import treeinterpreter as ti
except ImportError:
	pass

try:
	import xgboost as xgb
except ImportError:
	pass

try:
	from sknn.platform import gpu32
except ImportError:
	pass

try:
	from keras.wrappers.scikit_learn import KerasClassifier
	from keras.models import Sequential
except ImportError:
	pass

try:
	import lasagne
	from lasagne.layers import DenseLayer
	from lasagne.layers import InputLayer
	from lasagne.layers import DropoutLayer
	from lasagne.updates import adagrad, nesterov_momentum
	from lasagne.nonlinearities import softmax, tanh
	from lasagne.objectives import binary_crossentropy
	from nolearn.lasagne import NeuralNet
	from nolearn.lasagne import TrainSplit
except ImportError:
	pass

SEED = 42
np.random.seed(SEED)

def _predict_proba(cls, preproc, est, weight, x):
	proba = est.predict_proba(cls.applyPreproc(preproc, x))
	result = weight*proba
	return result

class MetaClassifierException(Exception):
	def __init__(self, e):
		self.exception = e
	def __str__(self):
		return self.exception


class MetaClassifier(BaseEstimator, ClassifierMixin):

	VALID_INTERP_ALGOS = [
		'RFC',
		'GBC'
	]

	def __init__(self, weights=[], parallel=False, verbose=False):
		super(self.__class__, self).__init__()
		self.__estimators = list()
		self.__weights = weights
		self.__parallel = parallel
		self.__verbose = verbose

		self.feature_importances_ = list()
		self.classes_ = list()

		self.labelBinarizer = LabelBinarizer()
		self.standardScaler = StandardScaler()

	def fit(self, X, y):
		X = X.astype(np.float32)
		y = y.astype(np.int32)

		self.standardScaler.fit(X)
		self.labelBinarizer.fit(y)
		self.classes_ = self.labelBinarizer.classes_

		for name, preproc, est in self.__estimators:
			if self.__verbose: print("Fitting estimator %s" % name)
			est.fit(self.applyPreproc(preproc, X), y)

		self.getFeatureImportance()

	def predict(self, x):
		probas = self.predict_proba(x)
		indices = probas.argmax(axis=1)
		return self.labelBinarizer.classes_[indices]

	def predict_proba(self, x):
		if self.__weights is None or not list(self.__weights):
			self.__weights = np.ones(len(self.__estimators))

		if len(self.__weights) != len(self.__estimators):
			raise MetaClassifierException("Number of weights to estimator mismatch!")

		predictions = list()
		weights = self.__weights/np.sum(self.__weights)

		if self.__parallel:
			from joblib import Parallel, delayed
			estList = list()
			for (name, preproc, est), weight in zip(self.__estimators, weights):
				estList.append((self, preproc, est, weight, x))

			with Parallel(n_jobs=len(self.__estimators), backend="threading") as parallel:
				predictions += parallel(
					delayed(_predict_proba)(*job) for job in estList
				)
		else:
			for (name, preproc, est), weight in zip(self.__estimators, weights):
				probs = est.predict_proba(self.applyPreproc(preproc, x))

				predictions.append(probs*weight)

		return np.sum(predictions, axis=0)

	def setWeights(self, weights):
		self.__weights = weights

	def setVerbose(self, verbose):
		self.__verbose = verbose

	def getFeatureImportance(self):
		self.feature_importances_ = list()
		for (name, preproc, est) in self.__estimators:
			if hasattr(est, 'feature_importances_'):
				self.feature_importances_.append(est.feature_importances_)
		if self.feature_importances_:
			self.feature_importances_ = np.mean(np.array(self.feature_importances_), axis=0)
		return self.feature_importances_

	def getTreeInterpretation(self, x):
		estList = self.getEstimatorList()
		prediction, bias, contributions = list(), list(), list()
		for (name, preproc, est) in estList:
			if name in MetaClassifier.VALID_INTERP_ALGOS:
				_x = self.applyPreproc(preproc, x)
				p, b, c = ti.predict(est, _x)
				prediction.append(p)
				bias.append(b)
				contributions.append(c)

		prediction = np.mean(np.array(prediction), axis=0)
		bias = np.mean(np.array(bias), axis=0)
		contributions = np.mean(np.array(contributions), axis=0)
		return prediction, bias, contributions

	def getEstimatorList(self):
		return self.__estimators

	def resetEstimatorList(self):
		self.__estimators = list()

	def applyPreproc(self, preproc, x):
		if type(x) is not np.array:
			x = np.array(x)

		if preproc == 'scale':
			if self.__verbose: print("preproc: StandardScaler")
			x_ = self.standardScaler.transform(x)
			return x_.astype(np.float32)
		if preproc:
			if self.__verbose: print("preproc:", preproc)
			x_ = np.copy(x)
			x_ = preproc(x_)
			return x_.astype(np.float32)
		else:
			return x.astype(np.float32)

	def addABC(self, preproc=None, params={}):
		name = 'ABC'
		self.getEstimatorList().append((name, preproc, AdaBoostClassifier(**params)))

	def addRFC(self, preproc=None, params={}):
		name = 'RFC'
		self.getEstimatorList().append((name, preproc, RandomForestClassifier(**params)))

	def addETC(self, preproc=None, params={}):
		name = 'ETC'
		self.getEstimatorList().append((name, preproc, ExtraTreesClassifier(**params)))

	def addLR(self, preproc=None, params={}):
		name = 'LR'
		self.getEstimatorList().append((name, preproc, LogisticRegression(**params)))

	def addGBC(self, preproc=None, params={}):
		name = 'GBC'
		self.getEstimatorList().append((name, preproc, GradientBoostingClassifier(**params)))

	def addXGBC(self, preproc=None, params={}):
		name = 'XGBC'
		self.getEstimatorList().append((name, preproc, xgb.XGBClassifier(**params)))

	def addKNC(self, preproc=None, params={}):
		name = 'KNC'
		self.getEstimatorList().append((name, preproc, KNeighborsClassifier(**params)))

	def addMLPC(self, preproc=None, params={}):
		name = 'MLPC'
		self.getEstimatorList().append((name, preproc, MLPClassifier(**params)))

	def addSVC(self, preproc=None, params={}):
		name = 'SVC'
		self.getEstimatorList().append((name, preproc, SVC(**params)))

	def addKNN(self, preproc=None, params={}):
		name = 'KNN'

		est = KerasClassifier(
			build_fn=params['build_fn'],
			nb_epoch=params['nb_epoch'],
			batch_size=64, #params['batch_size'],
			verbose=0
		)

		self.getEstimatorList().append((name, preproc, est))

	def addLNN(self, preproc=None, params={}):
		name = 'LNN'

		lasagne.random.set_rng(np.random.RandomState(SEED))

		layers = [
			('input', InputLayer),
			('dense0', DenseLayer),
			('dropout0', DropoutLayer),
			('dense1', DenseLayer),
			('dropout1', DropoutLayer),
			('dense2', DenseLayer),
			('dropout2', DropoutLayer),
			('output', DenseLayer)
		]
		input_shape = params['input_shape']
		est = NeuralNet(layers=layers,
			input_shape=(None, input_shape),
			dense0_num_units=params['dense0_num_units'],
			dense0_nonlinearity=tanh,
			dropout0_p=params['dropout0_p'],
			dense1_num_units=params['dense1_num_units'],
			dense1_nonlinearity=tanh,
			dropout1_p=params['dropout1_p'],
			dense2_num_units=params['dense2_num_units'],
			dense2_nonlinearity=tanh,
			dropout2_p=params['dropout2_p'],
			output_num_units=params['output_shape'],
			output_nonlinearity=softmax,
			update=adagrad,
			update_learning_rate=params['update_learning_rate'],
			#train_split=TrainSplit(params['train_split']),
			max_epochs=params['max_epochs'],
			verbose=1,
		)

		self.getEstimatorList().append((name, preproc, est))
