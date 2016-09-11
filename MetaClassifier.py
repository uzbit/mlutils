
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

def buildKeras():
	model = Sequential()
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	
	input_shape = 59
	model.add(Dense(1000,
		input_dim=input_shape,
		init='uniform', activation='tanh')
	)
	model.add(Dropout(0.5))
	model.add(Dense(500,
		init='uniform', activation='tanh')
	)
	model.add(Dropout(0.1))
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
		optimizer=sgd, metrics=['accuracy'],
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
			print X
			est.fit(X, y)

	def predict_proba(self, x):
		self.predictions_ = list()
		for name, preproc, est in self.estimators_:
			if preproc:
				x = preproc(x)
			
			self.predictions_.append(est.predict_proba(x))
		return np.mean(self.predictions_, axis=0)

class MetaClassifier(object):

	def __init__(self, params={}, normalize=False):
		self.__metaClf = None
		self.__params = params
		#self.__normalize = normalize
		#self.__normalizer = Normalizer()
		self.clfList = list()
		self.feature_importances_ = list()
	
	def train(self, X, y):
		self.__metaClf = EnsembleClassifier(
			estimators=self.clfList,
			#voting='soft'
		)
		X = X.astype(np.float32)
		y = y.astype(np.int32)
		
		#if self.__normalize:
		#	X = self.__normalizer.fit_transform(X)
			
		self.__metaClf.fit(X, y)
		#print self.__metaClf.estimators_

	def predict(self, x):
		#if self.__normalize:
		#	x = self.__normalizer.transform(x)
		return self.__metaClf.predict(x)
	
	def predict_proba(self, x):
		#for est in self.__metaClf.estimators_:
		#	print est
		#if self.__normalize:
		#	x = self.__normalizer.transform(x)
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
				'input_shape': 4,
				'output_num_units': 3,
				'train_split': 0.3,
			},
		}

	def getParams(self):
		return self.__params
	
	def addRF(self, preproc=None, params={}):
		name = 'RF'
		if not params:
			params = MetaClassifier.getDefaultParams()[name]
		self.__params[name].update(params)
		self.clfList.append((name, preproc, RandomForestClassifier(**self.__params[name])))

	def addLR(self, preproc=None, params={}):
		name = 'LR'
		if not params:
			params = MetaClassifier.getDefaultParams()[name]
		self.__params[name].update(params)
		self.clfList.append((name, preproc, LogisticRegression(**self.__params[name])))

	def addGBC(self, preproc=None, params={}):
		name = 'GBC'
		if not params:
			params = MetaClassifier.getDefaultParams()[name]
		self.__params[name].update(params)
		self.clfList.append((name, preproc, GradientBoostingClassifier(**self.__params[name])))

	def addXGBC(self, preproc=None, params={}):
		name = 'XGBC'
		if not params:
			params = MetaClassifier.getDefaultParams()[name]
		self.__params[name].update(params)
		self.clfList.append((name, preproc, xgboost.XGBClassifier(**self.__params[name])))

	def addKNC(self, preproc=None, params={}):
		name = 'KNC'
		if not params:
			params = MetaClassifier.getDefaultParams()[name]
		self.__params[name].update(params)
		self.clfList.append((name, preproc, KNeighborsClassifier(**self.__params[name])))
	
	#def addMLP(self, preproc=None, params={}):
	#	name = 'MLP'
	#	self.__params[name].update(params)
	#	self.clfList.append((name, preproc, KNeighborsClassifier(**self.__params[name])))
	
	def addBRBM(self, preproc=None, params={}):
		name = 'BRBM'
		if not params:
			params = MetaClassifier.getDefaultParams()[name]
		self.__params[name].update(params)
		self.clfList.append((name, preproc, BernoulliRBM(**self.__params[name])))
	
	def addKNN(self, preproc=None, params={}):
		name = 'KNN'		
		if not params:
			params = MetaClassifier.getDefaultParams()[name]
		self.__params[name].update(params)
		clf = FixedKerasClassifier(
			build_fn=buildKeras,
			nb_epoch=10,
			#batch_size=5,
			verbose=1
		)
		
		self.clfList.append((name, preproc, clf))

	def addLNN(self, preproc=None, params={}):
		name = 'LNN'		
		if not params:
			params = MetaClassifier.getDefaultParams()[name]
		self.__params[name].update(params)
		from nolearn.lasagne import NeuralNet
		from nolearn.lasagne import TrainSplit
		from lasagne.layers import DenseLayer
		from lasagne.layers import InputLayer
		from lasagne.layers import DropoutLayer
		from lasagne.updates import adagrad, nesterov_momentum
		from lasagne.nonlinearities import softmax, sigmoid, softplus
		from lasagne.objectives import binary_crossentropy
		from sklearn.pipeline import Pipeline
		
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
			('dense3', DenseLayer),
			('dropout3', DropoutLayer),
			('dense4', DenseLayer),
			('dropout4', DropoutLayer),
			#('dense5', DenseLayer),
			#('dropout1', DropoutLayer),
			('output', DenseLayer)
		]
		input_shape = self.__params[name]['input_shape']
		nn = NeuralNet(layers=layers,
			input_shape=(None, input_shape),
			dense0_num_units=input_shape*2,
			dropout0_p=0.4,
			#dense0_nonlinearity=softmax,
			dense1_num_units=input_shape*2,
			dropout1_p=0.4,
			dense2_num_units=input_shape*2,
			dropout2_p=0.4,
			dense3_num_units=input_shape*2,
			dropout3_p=0.4,
			dense4_num_units=input_shape*2,
			dropout4_p=0.4,
			output_num_units=2, #self.__params[name].update(params)['output_num_units'],
			output_nonlinearity=softmax,
			update=nesterov_momentum,
			update_learning_rate=0.001,
			update_momentum=0.9,
			#objective_loss_function=lasagne.objectives.binary_crossentropy,
			regression=False,
			train_split=TrainSplit(eval_size=self.__params[name]['train_split']),
			verbose=1,
			max_epochs=500
		)
		pipeline = Pipeline([
			('nn', nn),
		])
		
		self.clfList.append((name, preproc, pipeline))

	def __getFeatureImportance(self):
		for est in self.__metaClf.estimators_:
			if type(est) is xgboost.XGBClassifier:
				self.feature_importances_ = est.feature_importances_
	
