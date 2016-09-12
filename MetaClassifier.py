
#from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import BernoulliRBM
from sknn.platform import gpu32
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.local import LocallyConnected1D
from keras.optimizers import SGD

import numpy as np
import xgboost
np.random.seed(42)

def buildKeras():
	model = Sequential()
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	
	input_shape = 59
	model.add(Dense(1000,
		input_dim=input_shape,
		init='uniform', activation='tanh')
	)
	model.add(Dropout(0.5))
	model.add(Dense(500,
		init='uniform', activation='tanh')
	)
	model.add(Dropout(0.3))
	model.add(Dense(50,
		init='uniform', activation='tanh')
	)
	model.add(Dropout(0.1))
	model.add(Dense(1,
		init='uniform', activation='sigmoid')
	)
	# Compile model
	model.compile(
		loss='binary_crossentropy',
		optimizer='adagrad', metrics=['accuracy'],
	)
	return model

class FixedKerasClassifier(KerasClassifier):
	def predict_proba(self, X, **kwargs):
		kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
		probs = self.model.predict_proba(X, **kwargs)
		if(probs.shape[1] == 1):
			probs = np.hstack([1-probs,probs]) 
		return probs		


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, estimators=None):
		self.estimators_ = estimators

	def fit(self, X, y):
		for name, preproc, est in self.estimators_:
			if preproc:
				X = preproc(X)
			est.fit(X, y)

	def predict_proba(self, x):
		self.predictions_ = list()
		for name, preproc, est in self.estimators_:
			if preproc:
				x = preproc(x)
			
			self.predictions_.append(est.predict_proba(x))
		return np.mean(self.predictions_, axis=0)

class MetaClassifier(object):

	def __init__(self):
		self.__metaClf = None
		#self.__params = MetaClassifier.getDefaultParams()
		self.clfList = list()
		self.feature_importances_ = list()
	
	def train(self, X, y):
		self.__metaClf = EnsembleClassifier(
			estimators=self.clfList,
		)
		X = X.astype(np.float32)
		y = y.astype(np.int32)
		
		self.__metaClf.fit(X, y)
		
	def predict(self, x):
		return self.__metaClf.predict(x)
	
	def predict_proba(self, x):
		return self.__metaClf.predict_proba(x)
	
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
		self.clfList.append((name, preproc, RandomForestClassifier(**params)))

	def addLR(self, preproc=None, params={}):
		name = 'LR'
		self.clfList.append((name, preproc, LogisticRegression(**params)))

	def addGBC(self, preproc=None, params={}):
		name = 'GBC'
		self.clfList.append((name, preproc, GradientBoostingClassifier(**params)))

	def addXGBC(self, preproc=None, params={}):
		name = 'XGBC'
		self.clfList.append((name, preproc, xgboost.XGBClassifier(**params)))

	def addKNC(self, preproc=None, params={}):
		name = 'KNC'
		self.clfList.append((name, preproc, KNeighborsClassifier(**params)))
	
	#def addMLP(self, preproc=None, params={}):
	#	name = 'MLP'
	#	params.update(params)
	#	self.clfList.append((name, preproc, KNeighborsClassifier(**params)))
	
	def addBRBM(self, preproc=None, params={}):
		name = 'BRBM'
		self.clfList.append((name, preproc, BernoulliRBM(**params)))
	
	def addKNN(self, preproc=None, params={}):
		name = 'KNN'		
		clf = FixedKerasClassifier(
			build_fn=buildKeras,
			nb_epoch=10,
			#batch_size=5,
			verbose=1
		)
		
		self.clfList.append((name, preproc, clf))

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
		
		#lasagne.random.set_rng(np.random.RandomState(1))
		
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
		nn = NeuralNet(layers=layers,
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
			#objective = 
			train_split=TrainSplit(params['train_split']),
			verbose=1,
			max_epochs=params['max_epochs']
		)
		
		self.clfList.append((name, preproc, nn))
	

	def __getFeatureImportance(self):
		for est in self.__metaClf.estimators_:
			if type(est) is xgboost.XGBClassifier:
				self.feature_importances_ = est.feature_importances_
	
