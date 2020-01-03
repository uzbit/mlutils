import os
import random
import numpy as np
import pickle as pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,  matthews_corrcoef, make_scorer, roc_curve, auc
from sklearn.model_selection import train_test_split

from .MetaClassifier import MetaClassifier

LOGIT_ACCEPT_RATE = 0.5
SEED = 42

def plot_hist(y1, y2 = None, binFactor=50.0, title=''):
	import matplotlib.pyplot as plt
	thisMax = max(y1)
	thisMin = min(y1)
	if y2 is not None:
		max2 = max(y2)
		min2 = min(y2)
		thisMax = max(thisMax, max2)
		thisMin = min(thisMin, min2)

	thisWidth = (thisMax - thisMin)/binFactor
	try:
		plt.hist(y1, alpha = 0.5, bins=np.arange(thisMin, thisMax + thisWidth,  thisWidth), label='y1')
		if y2 is not None:
			plt.hist(y2, alpha = 0.5, bins=np.arange(thisMin, thisMax + thisWidth,  thisWidth), label='y2')
	except IndexError:
		print(title, 'had no values!')

	plt.title(title)
	plt.legend()
	plt.show()

def plot_importance(clf, columns):
	import matplotlib.pyplot as plt
	feature_importance = clf.feature_importances_
	# make importances relative to max importance
	feature_importance = 100.0 * (feature_importance / feature_importance.max())
	sorted_idx = np.argsort(feature_importance)
	pos = np.arange(sorted_idx.shape[0]) + .5
	plt.figure(figsize=(12, 6))
	plt.subplot(1, 1, 1)
	plt.barh(pos, feature_importance[sorted_idx], align='center')
	plt.yticks(pos, columns[sorted_idx])
	plt.xlabel('Relative Importance')
	plt.title('Variable Importance')
	plt.show()

def plot_deviance(clf, X, y, n_estimators):
	import matplotlib.pyplot as plt
	offset = int(X.shape[0] * 0.9)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]
	clf.fit(X_train, y_train)

	test_score = np.zeros((n_estimators,), dtype=np.float64)

	for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
		test_score[i] = clf.loss_(y_test, y_pred)

	plt.figure(figsize=(12, 6))
	#plt.subplot(1, 2, 1)
	plt.title('Deviance')
	plt.plot(np.arange(n_estimators) + 1, clf.train_score_, 'b-',
			 label='Training Set Deviance')
	plt.plot(np.arange(n_estimators) + 1, test_score, 'r-',
			 label='Test Set Deviance')
	plt.legend(loc='upper right')
	plt.xlabel('Boosting Iterations')
	plt.ylabel('Deviance')
	plt.show()

def get_classification(y, rate=0.5):
	return np.array([1 if x else 0 for x in y >= rate])

def get_labelencoder(column_values):
	le = LabelEncoder()
	le.fit(column_values)
	return le

def get_remove_features(df, featureColumns, N=4):
	removeList = []
	for feat in featureColumns:
		vals = df[feat].values
		nthtile = np.percentile(vals, np.arange(0, 100, N))
		nth0 = nthtile[0]
		countDiff = 0
		for nth in nthtile:
			if nth != nth0:
				countDiff += 1
		if countDiff == 0:
			removeList.append(feat)
	return removeList

def transform_column(le, df, column):
	df[column] = le.transform(df[column])

def do_evo_search(X, y,
	grid={}, scorer=None, cv=3,
	population_size=50, mutation_prob=0.3, #crossover_prob=0.5,
	generations_number=20, n_jobs=4,
	gridpickle='bestParams.pickle'):
	print("Performing evolutionary XGBoost search...")
	import xgboost as xgb
	from evolutionary_search import EvolutionaryAlgorithmSearchCV
	from sklearn.pipeline import Pipeline
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=SEED)
	print("Training on ", X_test.shape)

	if not grid:
		grid = dict()
		grid['xgb__learning_rate'] = [0.001, 0.05, 0.1]
		grid['xgb__max_depth'] = [3, 5, 10, 20]
		grid['xgb__gamma'] = [0, 1, 10]
		grid['xgb__subsample'] = [0.75, 1]
		grid['xgb__colsample_bytree'] = [0.75, 1]
		grid['xgb__min_child_weight'] = [1, 3, 5, 10]
		#grid['xgb__base_score'] = [0.1]
		grid['xgb__max_delta_step'] = [0, 1, 5]
		grid['xgb__n_estimators'] = [200, 500, 1000]
		grid['xgb__reg_lambda'] = [1, 10, 100]
		grid['xgb__reg_alpha'] = [1, 10, 100]
		grid['xgb__silent'] = [1]
		grid['xgb__objective'] = ['binary:logistic']
		#grid['pca__n_components'] = [50, 100, 200]

	if not scorer:
		scorer = make_scorer(scorer_auc, greater_is_better=True)

	pipeline = Pipeline(steps=[
		('xgb', xgb.XGBClassifier())
	])

	clf = EvolutionaryAlgorithmSearchCV(
		pipeline,
		grid,
		scoring=scorer,
		verbose=True,
		n_jobs=n_jobs,
		cv=cv,
		population_size=population_size,
		mutation_prob=mutation_prob,
		generations_number=generations_number,
	)

	if gridpickle and os.path.exists(gridpickle):
		bestParams = pickle.load(open(gridpickle, 'rb'))
	else:
		clf.fit(X_test, y_test)
		print("Best score", clf.best_score_)
		print("Best params", clf.best_params_)
		bestParams = {x.split('__')[1]:clf.best_params_[x] for x in clf.best_params_ if x.split('__')[0] == 'xgb'}
		pickle.dump(bestParams, open(gridpickle, 'wb'))

	print(bestParams)
	return bestParams

def do_xgboost_hyperopt_search(X, y, cv=3, maxEvals=10, testSize=0.2, seed=SEED):
	if os.path.exists('bestParams.pickle'):
		return pickle.load(open('bestParams.pickle', 'rb'))

	import xgboost as xgb
	from hyperopt import hp
	from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

	print("Performing hyperopt search...")
	intChoices = {
		'n_estimators': np.arange(300, 10000, dtype=int),
		'max_depth': np.arange(3, 100, dtype=int),
	}
	space = {
		'n_estimators' : hp.choice('n_estimators', intChoices['n_estimators']),
		'learning_rate' : hp.uniform('learning_rate', 0.0001, 0.01),
		'max_depth' : hp.choice('max_depth', intChoices['max_depth']),
		'min_child_weight' : hp.uniform('min_child_weight', 0, 20),
		'subsample' : hp.uniform('subsample', 0.6, 1),
		'gamma' : hp.uniform('gamma', 0.6, 1),
		'reg_alpha' : hp.uniform('reg_alpha', 0, 1),
		'reg_lambda' : hp.uniform('reg_lambda', 1, 100),
		'colsample_bytree' : hp.uniform('colsample_bytree', 0.6, 1),
		'objective':'binary:logistic',
		'silent' : 1
	}

	def score(params):
		results = list()
		print("Testing for ", params)
		for i in range(cv):
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=testSize, stratify=y, random_state=seed+i
			)
			print("Train shape", X_train.shape)
			clf = xgb.XGBClassifier(**params)
			clf.fit(X_train, y_train,
				eval_set=[(X_train, y_train), (X_test, y_test)],
				early_stopping_rounds = 100,
				eval_metric='auc'
			)
			probs = clf.predict_proba(X_test, ntree_limit=clf.best_iteration)[:,1]
			fpr, tpr, _ = roc_curve(y_test, probs, pos_label=1)
			results.append(auc(fpr, tpr))

		print("Outcomes: ", results)
		print("This score:", 1.0-np.mean(results))
		print()
		return {'loss': 1.0-np.mean(results), 'status': STATUS_OK}

	trials = Trials()
	bestParams = fmin(score, space,
		algo=tpe.suggest,
		trials=trials,
		max_evals=maxEvals
	)
	for intChoice in intChoices:
		bestParams[intChoice] = intChoices[intChoice][bestParams[intChoice]]

	print("Saving the best parameters: ", bestParams)

	pickle.dump(bestParams, open('bestParams.pickle', 'wb'))
	return bestParams

def do_lnn_hyperopt_search(X, y, cv=3, maxEvals=10, testSize=0.2, seed=SEED):

	from hyperopt import hp
	from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

	print("Performing LNN hyperopt search...")

	intParams = [
		'dense0_num_units',
		'dense1_num_units',
		'dense2_num_units',
		'max_epochs',
	]

	space = {
		'dense0_num_units' : hp.qloguniform('dense0_num_units', np.log(1e3), np.log(1e4), 1), #hp.choice('dense0_num_units', intChoices['dense0_num_units']),
		'dense1_num_units' : hp.qloguniform('dense1_num_units', np.log(1e2), np.log(1e3), 1), #hp.choice('dense1_num_units', intChoices['dense1_num_units']),
		'dense2_num_units' : hp.qloguniform('dense2_num_units', np.log(1e1), np.log(1e2), 1), #hp.choice('dense2_num_units', intChoices['dense2_num_units']),
		'update_learning_rate' : hp.loguniform('update_learning_rate', np.log(1e-4), np.log(1e-1)),
		'dropout0_p' : hp.uniform('dropout0_p', 0.1, 0.5),
		'dropout1_p' : hp.uniform('dropout1_p', 0.1, 0.5),
		'dropout2_p' : hp.uniform('dropout2_p', 0.1, 0.5),
		'max_epochs' : hp.qloguniform('max_epochs', np.log(5e1), np.log(1e2), 1), #hp.choice('max_epochs', intChoices['max_epochs']),
		'train_split' : hp.uniform('train_split', 0.199999, 0.2),
	}

	def score(params):
		results = list()
		print("Testing for ", params)
		params['input_shape'] = X.shape[1]
		params['output_shape'] = 2
		for param in intParams:
			params[param] = int(params[param])

		for i in range(cv):
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=testSize, stratify=y, random_state=seed+i
			)
			print("Train shape", X_train.shape)
			mcObj = MetaClassifier()
			mcObj.resetEstimatorList() # why is this fucking necessary?!
			mcObj.addLNN(
				preproc='scale',
				params=params
			)
			mcObj.fit(X_train, y_train)
			results.append(get_auc(mcObj, X_test, y_test))

		print("Outcomes: ", results)
		print("This score:", 1.0-np.mean(results))
		print()
		return {'loss': 1.0-np.mean(results), 'status': STATUS_OK}

	trials = Trials()
	bestParams = fmin(score, space,
		algo=tpe.suggest,
		trials=trials,
		max_evals=maxEvals,
		#rseed=None
	)
	for param in intParams:
		bestParams[param] = int(bestParams[param])

	bestParams['input_shape'] = X.shape[1]
	bestParams['output_shape'] = 2

	print("Saving the best parameters: ", bestParams)

	pickle.dump(bestParams, open('bestParams_lnn.pickle', 'wb'))
	return bestParams


def do_knn_hyperopt_search(X, y, cv=3, maxEvals=10, testSize=0.2, seed=SEED):

	from hyperopt import hp
	from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
	from keras.wrappers.scikit_learn import KerasClassifier
	from keras.models import Sequential
	from keras.layers import Dense, Dropout
	from keras.layers.local import LocallyConnected1D
	from keras.optimizers import SGD

	print("Performing KNN hyperopt search...")

	intParams = [
		'dense0_num_units',
		'dense1_num_units',
		'dense2_num_units',
		'nb_epoch',
		#'batch_size',
	]

	space = {
		'dense0_num_units' : hp.qloguniform('dense0_num_units', np.log(1e3), np.log(1e4), 1), #hp.choice('dense0_num_units', intChoices['dense0_num_units']),
		'dense1_num_units' : hp.qloguniform('dense1_num_units', np.log(1e2), np.log(1e3), 1), #hp.choice('dense1_num_units', intChoices['dense1_num_units']),
		'dense2_num_units' : hp.qloguniform('dense2_num_units', np.log(1e1), np.log(1e2), 1), #hp.choice('dense2_num_units', intChoices['dense2_num_units']),
		'update_learning_rate' : hp.loguniform('update_learning_rate', np.log(1e-4), np.log(1e-1)),
		'dropout0_p' : hp.uniform('dropout0_p', 0.1, 0.5),
		'dropout1_p' : hp.uniform('dropout1_p', 0.1, 0.5),
		'dropout2_p' : hp.uniform('dropout2_p', 0.1, 0.5),
		'nb_epoch' : hp.qloguniform('nb_epoch', np.log(5e1), np.log(1e2), 1), #hp.choice('max_epochs', intChoices['max_epochs']),
		#'batch_size' : hp.quniform('batch_size', 1, 32, 1),
	}

	def score(params):
		results = list()
		print("Testing for ", params)

		def build_fn():
			model = Sequential()
			sgd = SGD(lr=params['update_learning_rate'], decay=1e-6, momentum=0.9, nesterov=True)

			model.add(Dense(int(params['dense0_num_units']),
				input_dim=params['input_shape'] ,
				init='uniform', activation='tanh')
			)
			model.add(Dropout(params['dropout0_p']))
			model.add(Dense(int(params['dense1_num_units']),
				init='uniform', activation='tanh')
			)
			model.add(Dropout(params['dropout1_p']))
			model.add(Dense(int(params['dense2_num_units']),
				init='uniform', activation='tanh')
			)
			model.add(Dropout(params['dropout2_p']))
			model.add(Dense(int(params['output_shape']),
				init='uniform', activation='sigmoid')
			)
			# Compile model
			model.compile(
				loss='binary_crossentropy',
				optimizer='adagrad', metrics=['accuracy'],
			)
			return model

		params['input_shape'] = X.shape[1]
		params['output_shape'] = 1

		for param in intParams:
			params[param] = int(params[param])

		for i in range(cv):
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=testSize, stratify=y, random_state=seed+i
			)
			print("Train shape", X_train.shape)
			mcObj = MetaClassifier()
			mcObj.resetEstimatorList() # why is this fucking necessary?!
			mcObj.addKNN(
				preproc='scale',
				params={
					'build_fn': build_fn,
					'nb_epoch': int(params['nb_epoch']),
					#'batch_size': int(params['batch_size']),
				}
			)
			mcObj.fit(X_train, y_train)
			results.append(get_auc(mcObj, X_test, y_test))

		print("Outcomes: ", results)
		print("This score:", 1.0-np.mean(results))
		print()
		return {'loss': 1.0-np.mean(results), 'status': STATUS_OK}

	trials = Trials()
	bestParams = fmin(score, space,
		algo=tpe.suggest,
		trials=trials,
		max_evals=maxEvals,
	)
	for param in intParams:
		bestParams[param] = int(bestParams[param])

	bestParams['input_shape'] = X.shape[1]
	bestParams['output_shape'] = 1

	print("Saving the best parameters: ", bestParams)

	pickle.dump(bestParams, open('bestParams_knn.pickle', 'wb'))
	return bestParams

def do_bayes_search(X, y, cv=3, testSize=0.3):
	if os.path.exists('bestParams.pickle'):
		return pickle.load(open('bestParams.pickle', 'rb'))

	print("Performing Bayesian search...")
	from bayes_opt.bayesian_optimization import BayesianOptimization
	import warnings
	warnings.filterwarnings("ignore")
	def xgboostcv(
		eta,
		max_depth,
		num_round,
		gamma,
		subsample,
		max_delta_step,
		min_child_weight,
		colsample_bytree,
		rate_drop,
		skip_drop,
		reg_alpha,
		reg_lambda,
		seed=1234,
		scorer=None
	):
		param = {
			'eta':eta,
			'max_depth':int(round(max_depth)),
			'num_round':int(round(num_round)),
			'gamma':max(0, gamma),
			'subsample':max(0, subsample),
			'max_delta_step':max(0, max_delta_step),
			'min_child_weight':max(0, min_child_weight),
			'colsample_bytree':max(0, colsample_bytree),
			#'rate_drop': max(0, rate_drop),
			#'skip_drop': max(0, skip_drop),
			'reg_alpha': max(0, reg_alpha),
			'reg_lambda': max(0, reg_lambda),
			'silent':1,
			'objective':'binary:logistic',
			'nthread':4,
		}

		results = list()
		for i in range(cv):
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=testSize, stratify=y, random_state=seed+i
			)
			xg_train = xgb.DMatrix(X_train, label=y_train)
			xg_test = xgb.DMatrix(X_test, label=y_test)

			model = xgb.train(
				param,
				xg_train,
				param['num_round'],
				evals=[(xg_test, 'test')],
				feval=eval_auc,
				#early_stopping_rounds=EARLY_STOPPING
			)
			preds = model.predict(xg_test, ntree_limit=model.best_iteration)
			results.append(eval_auc(preds, xg_test)[1])

		print("Outcomes: ", results)
		return np.mean(results)

	xgboostBO = BayesianOptimization(
		xgboostcv,
		{
			'eta': (0.001, 0.5),
			#'learning_rate': (0.001, 0.5),
			'max_depth': (3, 50),
			'num_round': (100, 1000),
			'gamma': (0, 100),
			'reg_lambda': (0, 1000),
			'reg_alpha': (0., 1.),
			'subsample': (0.8, 1.0),
			'colsample_bytree': (0.8, 1.0),
			'max_delta_step': (0, 10),
			'min_child_weight': (1, 50),
			#'rate_drop': (0., 1.),
			#'skip_drop': (0.7, 1.),
		}
	)
	xgboostBO.maximize(init_points=20, restarts=15, n_iter=50)
	print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
	bestParams = xgboostBO.res['max']['max_params']
	bestParams['max_depth'] = int(round(bestParams['max_depth']))
	bestParams['num_round'] = int(round(bestParams['num_round']))
	pickle.dump(bestParams, open('bestParams.pickle', 'wb'))
	return bestParams

def do_random_search(X, y, nIter=3,
	gridpickle='bestParams.pickle'):

	from sklearn.grid_search import RandomizedSearchCV
	from scipy.stats import randint as sp_randint
	from scipy.stats import uniform as sp_uniform
	clf = xgb.XGBClassifier()

	grid = dict()
	grid['max_depth'] = sp_randint(3, 15)
	grid['learning_rate'] = sp_uniform(loc=0.001, scale=0.1)
	grid['n_estimators'] = sp_randint(100, 1500)
	grid['silent'] = [True]
	grid['objective'] = ['binary:logistic']
	grid['gamma'] = sp_randint(1, 100)
	grid['min_child_weight'] = sp_randint(0, 20)
	grid['max_delta_step'] = sp_randint(0, 10)
	grid['subsample'] = sp_uniform(loc=0.7, scale=0.29)
	grid['colsample_bytree'] = sp_uniform(loc=0.7, scale=0.29)
	grid['reg_alpha'] = sp_uniform(loc=0.0, scale=1.0)
	grid['reg_lambda'] = sp_uniform(loc=1, scale=99)

	def report(grid_scores):
		top_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
		for i, score in enumerate(top_scores):
			print(("Model with rank: {0}".format(i + 1)))
			print(("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				  score.mean_validation_score,
				  np.std(score.cv_validation_scores))))
			print(("Parameters: {0}".format(score.parameters)))
			print("")
		return top_scores[0]

	if gridpickle and os.path.exists(gridpickle):
		bestParams = pickle.load(open(gridpickle, 'rb'))
	else:
		randomSearchCV = RandomizedSearchCV(
			clf,
			cv=3,
			scoring='roc_auc',
			param_distributions=grid,
			n_iter=nIter,
			random_state=SEED,
			verbose=100
		)

		randomSearchCV.fit(X, y)
		report(randomSearchCV.grid_scores_)
		bestParams = randomSearchCV.best_params_
		print(bestParams)
		pickle.dump(bestParams, open(gridpickle, 'wb'))

	return bestParams

def print_feature_importance(model, cols):
	fmap = model.get_fscore()
	print("There are %d cols and only %d are used." % (len(cols), len(list(fmap.keys()))))
	sortedList = list()
	for feat, score in sorted(list(fmap.items()), key=lambda x: x[1], reverse=True):
		feat_idx = int(feat[1:])
		sortedList.append([feat_idx, fmap[feat], cols[feat_idx]])
		print(sortedList[-1])

	return sortedList

def print_confusion_matrix(label, preds, labels=None):
	cm = confusion_matrix(label, preds, labels=labels)
	print("confusion matrix:")
	print("label=class0, pred=class0", cm[0][0])
	print("label=class1, pred=class1", cm[1][1])
	print("label=class0, pred=class1", cm[0][1])
	print("label=class1, pred=class0", cm[1][0])
	print("Class0 True rate", cm[0][0]/float(cm[0][0]+cm[0][1]))
	print("Class1 True rate", cm[1][1]/float(cm[1][1]+cm[1][0]))
	print("Class0 False rate", cm[0][1]/float(cm[0][0]+cm[0][1]))
	print("Class1 False rate", cm[1][0]/float(cm[1][1]+cm[1][0]))

def get_confusion_rates(label, preds, labels=None):
	cm = confusion_matrix(label, preds, labels=labels)
	ret = {
		"class0true": cm[0][0]/float(cm[0][0]+cm[0][1]),
		"class1true": cm[1][1]/float(cm[1][1]+cm[1][0]),
		"class0false": cm[0][1]/float(cm[0][0]+cm[0][1]),
		"class1false": cm[1][0]/float(cm[1][1]+cm[1][0]),
	}
	return ret

def get_auc(clf, X_test, y_test):
	probs = clf.predict_proba(X_test)[:,1]
	fpr, tpr, _ = roc_curve(y_test, probs, pos_label=1)
	thisAUC = auc(fpr, tpr)
	return thisAUC

def scorer_auc(labels, preds):
	fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
	score = auc(fpr, tpr)
	return score

def eval_auc(preds, dtrain):
	labels = dtrain.get_label()
	fpr, tpr, _ = roc_curve(labels, preds, pos_label=1)
	score = auc(fpr, tpr)
	return 'auc', score

def eval_error(preds, dtrain):
	labels = dtrain.get_label()
	return 'error', float(sum(labels != (preds > RETURN_ACCEPT_RATE))) / len(labels)

def scorer_mcc(labels, preds):
	preds = get_classification(preds, rate=LOGIT_ACCEPT_RATE)
	coeff = matthews_corrcoef(labels, preds)
	return coeff

def eval_mcc(preds, dtrain):
	labels = dtrain.get_label()
	preds = get_classification(preds, rate=LOGIT_ACCEPT_RATE)
	coeff = matthews_corrcoef(labels, preds)
	return 'MCC', -coeff

def eval_custom(preds, dtrain):
	labels = dtrain.get_label()
	preds = get_classification(preds, rate=LOGIT_ACCEPT_RATE)
	cm = confusion_matrix(labels, preds)
	alpha = 1.0
	beta = 1.0
	if cm[1][1] > 0 and cm[0][0] > 0:
		pos = float(cm[1][0])/cm[1][1]
		neg = float(cm[0][1])/cm[0][0]
		score = 1. - alpha*pos - beta*neg + pos*neg*alpha*beta
	else:
		score = -(float(cm[0][1])+float(cm[1][0]))
	return 'custom', -score

def eval_custom2(preds, dtrain):
	labels = dtrain.get_label()
	preds = get_classification(preds, rate=LOGIT_ACCEPT_RATE)
	cm = confusion_matrix(labels, preds)
	if cm[0][1] > 0 and cm[1][1]+cm[1][0] > 0 and cm[0][0]+cm[0][1] > 0:
		tpRate = (cm[1][1]/float(cm[1][1]+cm[1][0]))
		fpRate = (cm[0][1]/float(cm[0][0]+cm[0][1]))
		score = tpRate/fpRate
	else:
		score = -(float(cm[0][1])+float(cm[1][0]))

	return 'custom2', -score
