# -*- coding: utf-8 -*-

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv
import hyperopt

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

print train.count()
print train.head()
# print train.describe()

# print train[['Pclass', 'Survived']].corr()

def calc_iv(feature, lable):
	data = pd.DataFrame({'feature': feature, 'label': lable, 'index': range(len(feature))})
	grouping = data.groupby(['feature', 'label'])['index'].count()
	print grouping
	
	feature_unique = np.unique(feature)
	positive_label, negative_label = np.ones(len(feature_unique)), np.zeros(len(feature_unique))

	positive_count, negative_count = float(len(lable[lable==1])), float(len(lable[lable==0]))
	# print positive_count, negative_count

	pyi = grouping[zip(feature_unique, positive_label)].values / positive_count
	pni = grouping[zip(feature_unique, negative_label)].values / negative_count
	woe = np.log(pyi / pni)
	iv = (pyi - pni) * woe
	return np.sum(iv)

def trans_2_titile(name):
	item = re.compile('[,.]\s+').split(name)[1]
	if item in ['Ms', 'Mme', 'Mlle']:
		return 'Mlle'
	elif item in ['Capt', 'Don', 'Major', 'Sir']:
		return 'Sir'
	elif item in ['Dona', 'Lady', 'the Countess', 'Jonkheer']:
		return 'Lady'
	else:
		return item

def ticket_count(ticket):
	ticket_group = ticket.groupby(ticket).count()
	share_ticket_name = ticket_group[ticket_group > 1].index
	return ['share' if item in share_ticket_name else 'unique' for item in ticket]

def predict_age(age_data):
	X_train_data = age_data[age_data['Age'].notna()]
	y_train = X_train_data['Age']
	X_train = X_train_data.drop(columns=['PassengerId', 'Age'])

	categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
	columns = X_train.columns.tolist()
	categorical_features_indices = [columns.index(item) for item in categorical_features]
	data_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)

	params = {'iterations': 50, 'eval_metric': 'RMSE', 'logging_level': 'Silent', 'l2_leaf_reg': 2.0, 'depth': 10, 'learning_rate': 0.007590561479761263}
	
	# def hyperopt_objective(space_params):
	# 	params.update(space_params)
	# 	model = CatBoostRegressor(**params)
	# 	cv_data = cv(data_pool, model.get_params())
	# 	best_auc = np.mean(cv_data['test-RMSE-mean'])
	# 	return -best_auc  # as hyperopt minimises

	# params_space = {
	# 	'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
	# 	'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
	# 	'depth': hyperopt.hp.randint('depth', 16)
	# }

	# trials = hyperopt.Trials()
	# best_param = hyperopt.fmin(hyperopt_objective, params_space, algo=hyperopt.tpe.suggest, max_evals=20, trials=trials)
	# print 'best_param:', best_param
	# params.update(best_param)

	# best_trials = min(trials.trials, key=lambda r: r['result']['loss'])
	# print 'trails for best params', best_trials

	reg = CatBoostRegressor(**params)
	reg.fit(X_train, y_train, cat_features=categorical_features_indices)

	X_predict_data = age_data[age_data['Age'].isna()]
	X_predict = X_predict_data.drop(columns=['PassengerId', 'Age'])
	y_predict = reg.predict(X_predict)

	return X_predict_data.index, y_predict

def handle_data(data):
	
	# print calc_iv(data['Pclass'], data['Survived'])

	data['Title'] = data['Name'].apply(trans_2_titile)
	# print calc_iv(data['Title'], data['Survived'])

	# print calc_iv(data['Sex'], data['Survived'])

	data['age_section'] = data['Age'].fillna(-1).apply(lambda item: int(item / 10) + 1)
	# print calc_iv(data['age_section'], data['Survived'])

	# print calc_iv(data['SibSp'], data['Survived'])

	# print calc_iv(data['Parch'], data['Survived'])

	data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
	# print calc_iv(data['FamilySize'], data['Survived'])

	data['TicketCount']  = ticket_count(data['Ticket'])
	# print calc_iv(data['TicketCount'], data['Survived'])


	data['Cabin'] = data['Cabin'].fillna('')
	data['Cabin_Area']  = data['Cabin'].apply(lambda item: item[0] if item != '' else 'X')
	# print calc_iv(data['Cabin_Area'], data['Survived'])

	data['Embarked'] = data['Embarked'].fillna('C')
	# print calc_iv(data['Embarked'], data['Survived'])

	if len(data[data['Age'].isna()]) != 0:
		row_indexer, age_value = predict_age(data[['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'Age']])
		data.loc[row_indexer, 'Age'] = age_value

	return data

train = handle_data(train)

print 'train'
print train.head()

def train_model(train_data):
	y_train = train_data['Survived']
	X_train = train_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'age_section'])

	categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'TicketCount', 'Cabin_Area']
	columns = X_train.columns.tolist()
	categorical_features_indices = [columns.index(item) for item in categorical_features]
	data_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)

	params = {'iterations': 50, 'eval_metric': 'AUC', 'logging_level': 'Silent', 'l2_leaf_reg': 1.0, 'depth': 4, 'learning_rate': 0.07877005965234678}
	# auc:0.8667860125253837
	
	# def hyperopt_objective(space_params):
	# 	params.update(space_params)
	# 	model = CatBoostClassifier(**params)
	# 	cv_data = cv(data_pool, model.get_params())
	# 	best_auc = np.mean(cv_data['test-AUC-mean'])
	# 	return -best_auc  # as hyperopt minimises

	# params_space = {
	# 	'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
	# 	'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
	# 	'depth': hyperopt.hp.randint('depth', 16)
	# }

	# trials = hyperopt.Trials()
	# best_param = hyperopt.fmin(hyperopt_objective, params_space, algo=hyperopt.tpe.suggest, max_evals=20, trials=trials)
	# print 'best_param:', best_param
	# params.update(best_param)

	# best_trials = min(trials.trials, key=lambda r: r['result']['loss'])
	# print 'trails for best params', best_trials

	clf = CatBoostClassifier(**params)
	clf.fit(X_train, y_train, cat_features=categorical_features_indices)

	joblib.dump(clf, './catboost_model.model')
	return clf

clf = train_model(train)
X_predict = handle_data(test)
X_predict = X_predict.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'age_section'])
y_predict = clf.predict(X_predict)

submit = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived': map(int, y_predict)})
submit.to_csv('./submission.csv', index=False)