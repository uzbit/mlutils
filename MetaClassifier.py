import numpy as np
from sknn.platform import gpu32
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.updates import adagrad, nesterov_momentum
from lasagne.nonlinearities import softmax, tanh
from lasagne.objectives import binary_crossentropy
import lasagne
import xgboost

SEED = 42
np.random.seed(SEED)

class MetaClassifierException(Exception):
	def __init__(self, e):
		self.exception = e
	def __str__(self):
		return self.exception

class MetaClassifier(object):

	def __init__(self, weights=list()):
		self.__estimators = list()
		self.__weights = weights
		self.feature_importances_ = list()
		self.standardScaler = StandardScaler()
		
	def fit(self, X, y):
		X = X.astype(np.float32)
		y = y.astype(np.int32)
		
		self.standardScaler.fit(X)

		for name, preproc, est in self.__estimators:
			est.fit(self.applyPreproc(preproc, X), y)

		self.getFeatureImportance()
		
	def predict_proba(self, x):
		if not list(self.__weights):
			self.__weights = np.ones(len(self.__estimators))
		
		if len(self.__weights) != len(self.__estimators):
			raise MetaClassifierException("Number of weights to estimator mismatch!")
		
		predictions = list()
		weights = self.__weights/np.sum(self.__weights)
		
		for (name, preproc, est), weight in zip(self.__estimators, weights):
			probs = est.predict_proba(self.applyPreproc(preproc, x))
			predictions.append(probs*weight)
		return np.sum(predictions, axis=0)
	
	def getEstimatorList(self):
		return self.__estimators

	def getFeatureImportance(self):
		featImportance = list()
		for (name, preproc, est) in self.__estimators:
			if hasattr(est, 'feature_importances_'):
				featImportance.append(est.feature_importances_)
		if featImportance:
			featImportance = np.mean(np.array(featImportance), axis=0)
		return featImportance

	def resetEstimatorList(self):
		self.__estimators = list()

	def applyPreproc(self, preproc, x):
		if preproc == 'scale':
			x_ = self.standardScaler.transform(x)
			return x_.astype(np.float32)
		if preproc:
			x_ = np.copy(x)
			x_ = preproc(x_)
			return x_.astype(np.float32)
		else:
			return x.astype(np.float32)
	
	@staticmethod
	def getDefaultParams():
		return {
			'RFC': {'random_state': SEED},
			'ETC': {'random_state': SEED},
			'LR': {},
			'GBC': {'random_state': SEED},
			'XGBC': {'seed': SEED},
			'MLP': {'random_state': SEED},
			'KNC': {'n_jobs':-1},
			'BRBM': {'random_state': SEED},
			'MLPC': {'random_state': SEED},
			'KNN': {},
			'LNN': {
				'dense0_num_units' : 1000,
				'dense1_num_units' : 500,
				'dense2_num_units' : 50,
				'dropout0_p' : 0.4,
				'dropout1_p' : 0.4,
				'dropout2_p' : 0.4,
				'max_epochs' : 40,
				'input_shape' : 59,
				'output_shape' : 2,
				'update_learning_rate' : 0.001,
				'train_split' : 0.2,
			},
		}

	def getParams(self):
		return self.__params
	
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
		self.getEstimatorList().append((name, preproc, xgboost.XGBClassifier(**params)))

	def addKNC(self, preproc=None, params={}):
		name = 'KNC'
		self.getEstimatorList().append((name, preproc, KNeighborsClassifier(**params)))
	
	def addBRBM(self, preproc=None, params={}):
		name = 'BRBM'
		self.getEstimatorList().append((name, preproc, BernoulliRBM(**params)))
	
	def addMLPC(self, preproc=None, params={}):
		name = 'MLPC'
		self.getEstimatorList().append((name, preproc, MLPClassifier(**params)))
	
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
	
		
