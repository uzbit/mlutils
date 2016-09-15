import numpy as np

from sknn.platform import gpu32
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import normalize
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential

import xgboost

np.random.seed(42)

class FixedKerasClassifier(KerasClassifier):
	def predict_proba(self, X, **kwargs):
		kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
		probs = self.model.predict_proba(X, **kwargs)
		if(probs.shape[1] == 1):
			probs = np.hstack([1-probs,probs]) 
		return probs		


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, estimators=list(), weights=list()):
		self.estimators_ = estimators
		self.weights_ = weights
		self.predictions_ = list()
		
	def fit(self, X, y):
		for name, preproc, est in self.estimators_:
			est.fit(MetaClassifier.applyPreproc(preproc, X), y)

	def predict_proba(self, x):
		self.predictions_ = list()
		if not self.weights_ or len(self.weights_) != len(self.estimators_):
			self.weights_ = np.ones(len(self.estimators_))
		weights = self.weights_/np.sum(self.weights_)
		for (name, preproc, est), weight in zip(self.estimators_, weights):
			predictions = est.predict_proba(MetaClassifier.applyPreproc(preproc, x))
			self.predictions_.append(predictions*weight)
		return np.sum(self.predictions_, axis=0)

class MetaClassifier(object):

	def __init__(self, weights=list()):
		self.__metaClf = EnsembleClassifier(weights=weights)
		self.feature_importances_ = list()
		
	def train(self, X, y):
		X = X.astype(np.float32)
		y = y.astype(np.int32)
		
		self.__metaClf.fit(X, y)
		
	def predict(self, x):
		return self.__metaClf.predict(x)
	
	def predict_proba(self, x):
		return self.__metaClf.predict_proba(x)
	
	def getEstimatorList(self):
		return self.__metaClf.estimators_
	
	def resetEstimatorList(self):
		self.__metaClf.estimators_ = list()
	
	@staticmethod
	def applyPreproc(preproc, x):
		if preproc:
			x_ = np.copy(x)
			return preproc(x_)
		else:
			return x
			
	@staticmethod
	def getDefaultParams():
		return {
			'RF': {'random_state': 42},
			'LR': {},
			'GBC': {'random_state': 42},
			'XGBC': {'seed': 42},
			'MLP': {'random_state': 42},
			'KNC': {'n_jobs':-1},
			'BRBM': {'random_state': 42},
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
	
	def addRF(self, preproc=None, params={}):
		name = 'RF'
		self.getEstimatorList().append((name, preproc, RandomForestClassifier(**params)))

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
	
	def addKNN(self, preproc=None, params={}):
		name = 'KNN'
		
		est = FixedKerasClassifier(
			build_fn=params['build_fn'],
			nb_epoch=params['nb_epoch'],
			batch_size=64, #params['batch_size'],
			verbose=0
		)
		
		self.getEstimatorList().append((name, preproc, est))

	def addLNN(self, preproc=None, params={}):
		name = 'LNN'		
		from nolearn.lasagne import NeuralNet
		from nolearn.lasagne import TrainSplit
		from lasagne.layers import DenseLayer
		from lasagne.layers import InputLayer
		from lasagne.layers import DropoutLayer
		from lasagne.updates import adagrad, nesterov_momentum
		from lasagne.nonlinearities import softmax, sigmoid, softplus, tanh
		from lasagne.objectives import binary_crossentropy
		import lasagne
		
		lasagne.random.set_rng(np.random.RandomState(1))
		
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
	

	def __getFeatureImportance(self):
		for est in self.__metaClf.estimators_:
			if type(est) is xgboost.XGBClassifier:
				self.feature_importances_ = est.feature_importances_
	
