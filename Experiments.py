from Utils import *
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from copy import deepcopy
import time

def LGMB_model(train_features,train_labels,valid_features,valid_labels):
	model = lgb.LGBMRegressor(n_estimators=2000, objective='regression', learning_rate=0.01, random_state=7)

	# Train the model
	model.fit(train_features, train_labels, eval_metric='rmse',early_stopping_rounds=50,
			  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
			  eval_names=['valid', 'train'], categorical_feature='auto')

	# Record the best score
	return model,model.best_score_['valid']['rmse'],model.best_score_['train']['rmse']

def CatBoost_model(train_features,train_labels,valid_features,valid_labels):
	model = CatBoostRegressor(
		iterations=150,
		depth=10,
		learning_rate=0.05,
		l2_leaf_reg=0.1,  # def=3
		loss_function='RMSE',
		eval_metric='RMSE',
		cat_features=train_features.select_dtypes('category').columns,
		random_strength=0.001,
		bootstrap_type='Bayesian',  # Poisson (supported for GPU only);Bayesian;Bernoulli;No
		bagging_temperature=1,  # for Bayesian bootstrap_type; 1=exp;0=1
		leaf_estimation_method='Newton',  # Gradient;Newton
		leaf_estimation_iterations=2,
		boosting_type='Ordered'  # Ordered-small data sets; Plain
		, feature_border_type='Median'  # Median;Uniform;UniformAndQuantiles;MaxLogSum;MinEntropy;GreedyLogSum
		, random_seed=7)

	model.fit(train_features, train_labels, use_best_model=True,
			  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
			  metric_period=10)

	# Record the best score
	return model,model.best_score_['validation_0']['RMSE'],model.best_score_['validation_1']['RMSE']


def new_features(data):
	data['vote_count_avg'] = data['vote_count']/(np.log2(data['passed_years']+1)+1)
	secret_weapon = pd.read_csv('./data/inflation_data.csv')
	secret_weapon = secret_weapon.set_index('year')
	secret_weapon['amount'] = secret_weapon['amount'].max()/secret_weapon['amount']
	for idx in data.index:
		data['budget'].loc[idx] = data['budget'].loc[idx]* secret_weapon.loc[data['release_year'].loc[idx], 'amount']

	data['avg_salary']=data['budget']/(data['num_crew']+data['num_cast']+1)
	data['budget_to_popularity'] = data['budget'] / (data['popularity']+1)
	data['budget_to_runtime'] = data['budget'] / (data['runtime']+1)
	data['runtime_to_mean_year'] = data['runtime'] / data.groupby("release_year")["runtime"].transform('mean')
	data['popularity_to_mean_year'] = data['popularity'] / data.groupby("release_year")["popularity"].transform('mean')
	data['budget_to_mean_year'] = data['budget'] / (data.groupby("release_year")["budget"].transform('mean')+1)

	return data

def train_model(X, y, folds, params=None, model_type='lgb', plot_feature_importance=False, model=None):
	n_fold = 10
	# folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

	oof = np.zeros(X.shape[0])
	scores = []
	feature_importance = np.zeros(X.shape[1])
	for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
		print('Fold', fold_n, 'started at', time.ctime())
		if model_type == 'sklearn':
			X_train, X_valid = X[train_index], X[valid_index]
		else:
			X_train, X_valid = X.loc[train_index], X.iloc[valid_index]
		y_train, y_valid = y[train_index], y[valid_index]

		if model_type == 'lgb':
			model = lgb.LGBMRegressor(**params, n_estimators=5000)
			model.fit(X_train, y_train,
					  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
					  early_stopping_rounds=200, categorical_feature='auto',verbose=100)
			y_pred_valid = model.predict(X_valid)

		if model_type == 'sklearn':
			model = model
			model.fit(X_train, y_train)
			y_pred_valid = model.predict(X_valid).reshape(-1, )
			score = mean_squared_error(y_valid, y_pred_valid)
			print("mean_squared_error: ", score)


		if model_type == 'cat':
			model = CatBoostRegressor(iterations=200, eval_metric='RMSE', **params)
			model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True,verbose=False)

			y_pred_valid = model.predict(X_valid)

		oof[valid_index] = y_pred_valid.reshape(-1, )
		scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)

		if model_type == 'lgb':
			feature_importance += model.feature_importances_

	print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

	if model_type == 'lgb':
		feature_importance /= n_fold
		if plot_feature_importance:
			plot_df = pd.DataFrame([[i[0], i[1]] for i in zip(feature_importance, X.columns)],columns=['importance', 'feature'])
			plot_feature_importances(plot_df)

			return model, np.mean(scores), feature_importance
		return model, np.mean(scores)

	else:
		return model, np.mean(scores)


def experiment_1(data_orig,target,model_name):
	data_orig = data_orig.drop(['imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue',
								'production_countries','spoken_languages','Keywords'], axis=1)
	for col in data_orig.columns:
		if data_orig[col].nunique() == 1:
			print(col)
			data_orig = data_orig.drop([col], axis=1)

	texts = data_orig[['title', 'overview', 'original_title']]

	for col in ['title', 'overview', 'original_title']:
		data_orig['len_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x)))
		data_orig['words_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x.split(' '))))
		data_orig = data_orig.drop(col, axis=1)

	X = data_orig.drop(['id', 'revenue'], axis=1)
	y = np.log10(data_orig['revenue']+1)

	n_fold = 7
	folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
	oof_texts = []
	for col in texts.columns:
		vectorizer = TfidfVectorizer(
			sublinear_tf=True,
			analyzer='word',
			token_pattern=r'\w{1,}',
			ngram_range=(1, 2),
			min_df=10
		)
		vectorizer.fit(list(texts[col].fillna('')) + list(texts[col].fillna('')))
		train_col_text = vectorizer.transform(texts[col].fillna(''))
		model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)
		oof_text, _ = train_model(train_col_text, y, folds=folds, params=None, model_type='sklearn',
												model=model)
		oof_texts.append(oof_text)

		X[col + '_oof'] = oof_text.predict(train_col_text)

	X = new_features(X)
	numeric_cols =  ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count','num_crew',
						  'num_cast','num_companies','avg_salary', 'release_year',
						  'vote_count_avg', 'passed_years','budget_to_popularity','budget_to_runtime','runtime_to_mean_year',
					 'popularity_to_mean_year','budget_to_mean_year'] + [col for col in X.columns if '_oof' in col or 'len_' in col
																		 or 'words_' in col or 'log_' in col]
	cat_cols = list(set(X.columns)-set(numeric_cols))
	X = shrink_memory_consumption(X,cat_vals=cat_cols,numerical_vals=numeric_cols)

	n_fold = 7
	folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

	#building & blending

	if model_name=='LGBM':
		params = {'num_leaves': 30,
				  'min_data_in_leaf': 20,
				  'objective': 'regression',
				  'max_depth': 9,
				  'learning_rate': 0.01,
				  "boosting": "gbdt",
				  "feature_fraction": 0.9,
				  "bagging_freq": 1,
				  "bagging_fraction": 0.9,
				  "bagging_seed": 11,
				  "metric": 'rmse',
				  "lambda_l1": 0.2,
				  "verbosity": -1}
		model, predictions, _ = train_model(X, y, params=params, folds=folds, model_type='lgb',plot_feature_importance=True)
	elif model_name == 'CATBOOST':
		cat_params = {'learning_rate': 0.05,
					  'depth': 10,
					  'l2_leaf_reg': 0.1,
					  'loss_function': 'RMSE',
					  'colsample_bylevel': 0.8,
					  'bagging_temperature': 0.2,
					  'cat_features' : X.select_dtypes('category').columns,
					  'od_type': 'Iter',
					  'od_wait': 100,
					  'random_seed': 11,
					  'allow_writing_files': False}
		model, predictions = train_model(X, y, params=cat_params, folds=folds, model_type='cat',plot_feature_importance=True)

	return oof_texts, model, (10 ** np.mean(predictions))



def experiment_2(data_orig,target):
	print('--- starting training ----')
	data_orig = data_orig.drop(['imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue',
								'production_countries','spoken_languages','Keywords'], axis=1)

	texts = data_orig[['title', 'overview', 'original_title']]

	for col in ['title', 'overview', 'original_title']:
		data_orig['len_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x)))
		data_orig['words_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x.split(' '))))
		data_orig = data_orig.drop(col, axis=1)

	X = data_orig.drop(['id', target], axis=1)
	y = np.log10(data_orig[target]+1)

	oof_texts = []
	ridge_models = []
	for col in texts.columns:
		vectorizer = TfidfVectorizer(
			sublinear_tf=True,
			analyzer='word',
			token_pattern=r'\w{1,}',
			ngram_range=(1, 2),
			min_df=10
		)
		oof_text = vectorizer.fit(list(texts[col].fillna('')) + list(texts[col].fillna('')))
		train_col_text = vectorizer.transform(texts[col].fillna(''))
		model = linear_model.Ridge(0.1)
		ridge_model = model.fit(train_col_text, y)
		ridge_models.append(ridge_model)
		oof_texts.append(oof_text)
		X[col + '_oof'] = model.predict(train_col_text)

	numeric_cols =  ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count','num_crew',
						  'num_cast','num_companies','avg_salary', 'release_year',
						  'vote_count_avg', 'passed_years','budget_to_popularity','budget_to_runtime','runtime_to_mean_year',
					 'popularity_to_mean_year','budget_to_mean_year'] + [col for col in X.columns if '_oof' in col or 'len_' in col
																		 or 'words_' in col or 'log_' in col]
	cat_cols = list(set(X.columns)-set(numeric_cols))
	X = new_features(X)
	X = build_features(X,cat_cols,numeric_cols)

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

	#building & blending
	fit_params = {"early_stopping_rounds": 200,
				  "eval_metric": 'rmse',
				  "eval_set": [(X_test, y_test)],
				  'eval_names': ['valid'],
				  # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
				  'categorical_feature': 'auto'}

	from scipy.stats import randint as sp_randint
	from scipy.stats import uniform as sp_uniform
	param_test = {'num_leaves': sp_randint(6, 50),
				  'min_child_samples': sp_randint(0, 500),
				  'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
				  'subsample': sp_uniform(loc=0.2, scale=0.8),
				  'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
				  'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10],
				  'reg_lambda': [0, 1e-1, 1, 5, 10, 20]}

	clf = lgb.LGBMRegressor(max_depth=-1, random_state=314, silent=True,objective='regression', n_jobs=4, n_estimators=5000)
	grid = RandomizedSearchCV(
		estimator=clf, param_distributions=param_test,
		scoring='neg_root_mean_squared_error',
		cv=5,
		refit=True,
		random_state=314,
		verbose=True)

	grid.fit(X_train,y_train,**fit_params)

	# Results from Grid Search
	print("\n========================================================")
	print(" Results from Grid Search ")
	print("========================================================")

	# print("\n The best estimator across ALL searched params:\n",
	# 	  grid.best_estimator_)

	print("\n The best score across ALL searched params:\n",
		  10**grid.best_score_)

	print("\n The best parameters across ALL searched params:\n",
		  grid.best_params_)

	print("\n ========================================================")

	features_imp = grid.best_estimator_.feature_importances_
	predictors = [x for x in X_train.columns]

	plot_df = pd.DataFrame([[i[0], i[1]] for i in zip(features_imp, predictors)], columns=['importance', 'feature'])
	plot_feature_importances(plot_df)




def experiment_3(data_orig,target,model_name,plot=False):
	data_orig = data_orig.drop(['imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue',
								'production_countries','spoken_languages','Keywords'], axis=1)
	for col in data_orig.columns:
		if data_orig[col].nunique() == 1:
			print(col)
			data_orig = data_orig.drop([col], axis=1)

	texts = data_orig[['title', 'overview', 'original_title']]

	for col in ['title', 'overview', 'original_title']:
		data_orig['len_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x)))
		data_orig['words_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x.split(' '))))
		data_orig = data_orig.drop(col, axis=1)

	X = data_orig.drop(['id', target], axis=1)
	y = np.log10(data_orig[target]+1)

	n_fold = 7
	folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
	oof_texts = []
	for col in texts.columns:
		vectorizer = TfidfVectorizer(
			sublinear_tf=True,
			analyzer='word',
			token_pattern=r'\w{1,}',
			ngram_range=(1, 2),
			min_df=10
		)
		vectorizer.fit(list(texts[col].fillna('')) + list(texts[col].fillna('')))
		train_col_text = vectorizer.transform(texts[col].fillna(''))
		model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)
		oof_text, _ = train_model(train_col_text, y, folds=folds, params=None, model_type='sklearn',
												model=model)
		oof_texts.append(oof_text)
		X[col + '_oof'] = oof_text.predict(train_col_text)

	numeric_cols =  ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count','num_crew',
						  'num_cast','num_companies','avg_salary', 'release_year',
						  'vote_count_avg', 'passed_years','budget_to_popularity','budget_to_runtime','runtime_to_mean_year',
					 'popularity_to_mean_year','budget_to_mean_year'] + [col for col in X.columns if '_oof' in col or 'len_' in col
																		 or 'words_' in col or 'log_' in col]
	cat_cols = list(set(X.columns)-set(numeric_cols))
	X = new_features(X)
	X = build_features(X,cat_cols,numeric_cols)


	n_fold = 7
	folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

	#building & blending

	if model_name=='LGBM':
		params = {'num_leaves': 30,
				  'min_data_in_leaf': 20,
				  'objective': 'regression',
				  'max_depth': 9,
				  'learning_rate': 0.01,
				  "boosting": "gbdt",
				  "feature_fraction": 0.9,
				  "bagging_freq": 1,
				  "bagging_fraction": 0.9,
				  "bagging_seed": 11,
				  "metric": 'rmse',
				  "lambda_l1": 0.2,
				  "verbosity": -1}
		model, predictions, _ = train_model(X, y, params=params, folds=folds, model_type='lgb',
												 plot_feature_importance=True)
	elif model_name == 'CATBOOST':
		cat_params = {'learning_rate': 0.05,
					  'depth': 10,
					  'l2_leaf_reg': 0.1,
					  'loss_function': 'RMSE',
					  'colsample_bylevel': 0.8,
					  'bagging_temperature': 0.2,
					  'cat_features' : X.select_dtypes('category').columns,
					  'od_type': 'Iter',
					  'od_wait': 100,
					  'random_seed': 11,
					  'allow_writing_files': False}
		model, predictions = train_model(X, y, params=cat_params, folds=folds, model_type='cat')

	return (oof_texts, model,(10 ** np.mean(predictions)))

def train(data_orig,target,model_name,plot=False):
	print('--- starting training ----')
	data_orig = data_orig.drop(['imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue',
								'production_countries','spoken_languages','Keywords'], axis=1)

	texts = data_orig[['title', 'overview', 'original_title']]

	for col in ['title', 'overview', 'original_title']:
		data_orig['len_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x)))
		data_orig['words_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x.split(' '))))
		data_orig = data_orig.drop(col, axis=1)

	X = data_orig.drop(['id', target], axis=1)
	y = np.log10(data_orig[target]+1)

	oof_texts = []
	ridge_models = []
	for col in texts.columns:
		vectorizer = TfidfVectorizer(
			sublinear_tf=True,
			analyzer='word',
			token_pattern=r'\w{1,}',
			ngram_range=(1, 2),
			min_df=10
		)
		oof_text = vectorizer.fit(list(texts[col].fillna('')) + list(texts[col].fillna('')))
		train_col_text = vectorizer.transform(texts[col].fillna(''))
		model = linear_model.Ridge(0.1)
		ridge_model = model.fit(train_col_text, y)
		ridge_models.append(ridge_model)
		oof_texts.append(oof_text)
		X[col + '_oof'] = model.predict(train_col_text)

	numeric_cols =  ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count','num_crew',
						  'num_cast','num_companies','avg_salary', 'release_year',
						  'vote_count_avg', 'passed_years','budget_to_popularity','budget_to_runtime','runtime_to_mean_year',
					 'popularity_to_mean_year','budget_to_mean_year'] + [col for col in X.columns if '_oof' in col or 'len_' in col
																		 or 'words_' in col or 'log_' in col]
	cat_cols = list(set(X.columns)-set(numeric_cols))
	X = new_features(X)
	X = build_features(X,cat_cols,numeric_cols)
	X[target] = data_orig[target]
	print(X.shape)
	X,_ = delete_highly_target_correlated_features(X,target,threshold=0.8)
	X.drop([target],axis=1,inplace=True)
	print(X.shape)

	#building & blending

	if model_name=='LGBM':
		params = {'num_leaves': 36,
				  'min_child_samples': 22,
				  'objective': 'regression',
				  'max_depth': 9,
				  'learning_rate': 0.01,
				  "boosting": "gbdt",
				  "feature_fraction": 0.9,
				  "bagging_freq": 1,
				  "bagging_fraction": 0.9,
				  "bagging_seed": 11,
				  "metric": 'rmse',
				  "lambda_l1": 0.2}
		model = lgb.LGBMRegressor(**params, n_estimators=2000)
		model = model.fit(X,y, categorical_feature='auto')

	elif model_name == 'CATBOOST':
		cat_params = {'learning_rate': 0.05,
					  'l2_leaf_reg': 0.1,
					  'loss_function': 'RMSE',
					  'colsample_bylevel': 0.8,
					  'bagging_temperature': 0.2,
					  'cat_features' : X.select_dtypes('category').columns,
					  'od_type': 'Iter',
					  'od_wait': 100,
					  'random_seed': 11,
					  'allow_writing_files': False}
		model = CatBoostRegressor(iterations=200, eval_metric='RMSE', **cat_params)
		model = model.fit(X,y, use_best_model=True, verbose=False)

	# features_imp = model.feature_importances_
	# predictors = [x for x in X.columns]
	#
	# plot_df = pd.DataFrame([[i[0], i[1]] for i in zip(features_imp, predictors)], columns=['importance', 'feature'])
	# plot_feature_importances(plot_df)

	print('--- done training ----')
	return (model,ridge_models,oof_texts,data_orig.columns,X)



def enrich_features_with(features,df_built,args):
	rich_dfs = [[None,arg] for arg in args]
	for rich_df in rich_dfs:
		rich_df[0] = enrich_dataset_by_dividing_x(features, rich_df[1])
		rich_df[0] = build_features(rich_df[0], group_by_x=rich_df[1])
		rich_df[0] = rich_df[0].groupby('id').agg(['mean'])
		rich_df[0].columns = [col[0] for col in rich_df[0]]
		relevant_columns = [idx for idx, col in enumerate(rich_df[0].columns) if rich_df[1] in col]
		rich_df[0] = rich_df[0].iloc[:, relevant_columns]
		df_built = df_built.merge(rich_df[0], on='id', how='left')

	return df_built


def plot_feature_importances(df):
	"""
	Plot importances returned by a model. This can work with any measure of
	feature importance provided that higher importance is better.

	Args:
		df (dataframe): feature importances. Must have the features in a column
		called `features` and the importances in a column called `importance

	Returns:
		shows a plot of the 15 most importance features

		df (dataframe): feature importances sorted by importance (highest to lowest)
		with a column for normalized importance
		"""

	# Sort features according to importance
	df = df.sort_values('importance', ascending=False).reset_index()

	# Normalize the feature importances to add up to one
	df['importance_normalized'] = df['importance'] / df['importance'].sum()

	# Make a horizontal bar chart of feature importances
	plt.figure(figsize=(10, 6))
	ax = plt.subplot()

	# Need to reverse the index to plot most important on top
	ax.barh(list(reversed(list(df.index[:20]))),
			df['importance_normalized'].head(20),
			align='center', edgecolor='k')

	# Set the yticks and labels
	ax.set_yticks(list(reversed(list(df.index[:20]))))
	ax.set_yticklabels(df['feature'].head(20))

	# Plot labeling
	plt.xlabel('Normalized Importance');
	plt.title('Feature Importances')
	plt.show()

	return df

def predict(data_orig,target,models):
	print('--- starting predicting ----')
	data_orig = preprocess(data_orig,save=False,cols=models[-2])
	data_orig = data_orig.drop(['imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue',
								'production_countries','spoken_languages','Keywords'], axis=1)

	texts = data_orig[['title', 'overview', 'original_title']]
	for col in ['title', 'overview', 'original_title']:
		data_orig['len_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x)))
		data_orig['words_' + col] = data_orig[col].fillna('').apply(lambda x: len(str(x.split(' '))))
		data_orig = data_orig.drop(col, axis=1)

	X = data_orig.drop(['id', target], axis=1)
	y = np.log10(data_orig[target]+1)

	oof_texts = []
	for idx,col in enumerate(texts.columns):
		vectorizer = models[2][idx]
		train_col_text = vectorizer.transform(texts[col].fillna(''))
		model = models[1][idx]
		oof_text = model.fit(train_col_text, y)
		oof_texts.append(oof_text)
		X[col + '_oof'] = oof_text.predict(train_col_text)

	numeric_cols =  ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count','num_crew',
						  'num_cast','num_companies','avg_salary', 'release_year',
						  'vote_count_avg', 'passed_years','budget_to_popularity','budget_to_runtime','runtime_to_mean_year',
					 'popularity_to_mean_year','budget_to_mean_year'] + [col for col in X.columns if '_oof' in col or 'len_' in col
																		 or 'words_' in col or 'log_' in col]
	cat_cols = list(set(X.columns)-set(numeric_cols))
	X = new_features(X)
	X = build_features(X,cat_cols,numeric_cols)
	train_features = models[-1]
	cols_to_drop = [col for col in X.columns if col not in train_features.columns]
	X.drop(cols_to_drop,inplace=True,axis=1)

	group_by_variables = ['release_season','collection_id','original_language','jobs_Executive Producer','jobs_Art Direction']
	for idx in X.index:
		for var in group_by_variables:
			relevant_columns = [_ for _, col in enumerate(X.columns) if str(col).startswith(var) and col != var]
			if X.loc[idx, var] in list(train_features[var]):
				idx2 = train_features.index[train_features[var] == X.loc[idx, var]][0]
				X.iloc[idx, relevant_columns] = train_features.iloc[idx2, relevant_columns]

	model = models[0]
	predictions = model.predict(X)
	print('--- done predicting ----')
	return np.array(list(map(lambda x : 10**x ,predictions)))


def rmsle(y_true, y_pred):
	"""
	Calculates Root Mean Squared Logarithmic Error between two input vectors
	:param y_true: 1-d array, ground truth vector
	:param y_pred: 1-d array, prediction vector
	:return: float, RMSLE score between two input vectors
	"""
	assert y_true.shape == y_pred.shape, \
		ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
	return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

if __name__=='__main__':
	# raw_data = pd.read_csv('data/train.tsv', sep="\t")
	# df = preprocess(raw_data,train=True)
	df = pd.read_csv('./data/clean_data.csv')

	# models = ['LGBM','CATBOOST']
	# final_results =[]
	# for model in models:
	# 	result = experiment_1(df,'revenue',model)
	# 	# result = experiment_3(df, 'revenue',model)
	# 	final_results.append(result)
	# print(final_results[0][-1],final_results[1][-1])
	# pickle.dump(final_results,open('./models.pkl','wb'))

	# experiment_2(df,'revenue')
	results = train(df,'revenue','LGBM')
	pickle.dump(results,open('./models.pkl','wb'))
	results = pickle.load(open('./models.pkl','rb'))
	test_df = pd.read_csv('data/test.tsv', sep="\t")
	ground_truth = test_df['revenue'].values.flatten()
	predictions = predict(test_df,'revenue',results)
	print(rmsle(ground_truth,predictions))