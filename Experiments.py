from Utils import *
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from copy import deepcopy

def LGMB_model(train_features,train_labels,valid_features,valid_labels):
	model = lgb.LGBMRegressor(n_estimators=200, objective='regression', learning_rate=0.05,
							  reg_alpha=0.1, reg_lambda=0.1, random_state=7, importance_type='gain')

	# Train the model
	model.fit(train_features, train_labels, eval_metric='rmse',
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

def experiment_1(data_orig,target,model_name):
	# Create the kfold object
	number_of_folds = 7
	k_fold = KFold(n_splits=number_of_folds, shuffle=True, random_state=50)
	valid_scores = []
	train_scores = []
	predictors = []
	features_imp = 'empty'
	model=None
	data = deepcopy(data_orig)

	for column in list(data.select_dtypes('number').columns):
		if abs(data[column].skew()) > 1.0:
			data[column] = data[column].apply(lambda x: np.log10(x + 1))

	# Iterate through each fold
	for train_indices, valid_indices in k_fold.split(data):
		train_features, train_labels = data.loc[train_indices], data[target].loc[train_indices]
		valid_features, valid_labels = data.loc[valid_indices], data[target].loc[valid_indices]


		for df in [train_features,valid_features]:
			# date related features
			df['release_date'] = pd.to_datetime(df['release_date'])
			df['release_year'] = df['release_date'].dt.year
			df['release_month'] = df['release_date'].dt.month
			df['passed_years'] = date.today().year - df['release_year']
			df['release_season'] = pd.cut(df['release_month'], bins=[0, 3, 6, 9, 12],
										  labels=["Winter", "Spring", "Summer", "Autumn"]).astype('category')
			df.drop(['release_date', 'release_month','revenue','id'], axis=1,inplace=True)

		cat_vars = ['homepage', 'original_language', 'video', 'production_company_country', 'director_id','tagline','got_img',
					'producer_id', 'collection_id', 'production_company_id', 'genres', 'top_3_actors']\
				   +[col for col in train_features.columns if 'genre_' in col]
		numerical_vars = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count','crew_involved',
						  'actors_involved','production_companies_involved','avg_salary']
		train_features = shrink_memory_consumption(train_features,cat_vars,numerical_vars)
		valid_features = shrink_memory_consumption(valid_features,cat_vars,numerical_vars)
		if isinstance(features_imp,str):
			features_imp = np.zeros((valid_features.shape[1],))

		#Create the model
		if model_name=='LGBM':
			model, valid_score, train_score = LGMB_model(train_features,train_labels,valid_features,valid_labels)
		elif model_name=='CATBOOST':
			model, valid_score, train_score = CatBoost_model(train_features, train_labels, valid_features, valid_labels)


		valid_scores.append(valid_score)
		train_scores.append(train_score)

		features_imp += model.feature_importances_
		predictors = [x for x in train_features.columns]

	print('Avg train loss is {} million dollars'.format((10**np.mean(train_scores))))
	print('Avg valid loss is {} million dollars'.format((10**np.mean(valid_scores))))

	features_imp /= number_of_folds
	plot_df = pd.DataFrame([[i[0], i[1]] for i in zip(features_imp, predictors)], columns=['importance', 'feature'])
	plot_feature_importances(plot_df)

	return (10 ** np.mean(train_scores)), (10 ** np.mean(valid_scores))





def experiment_2(df,target):
	features_imp = 'empty'

	for column in list(df.select_dtypes('number').columns):
		if abs(df[column].skew()) > 1.0:
			df[column] = df[column].apply(lambda x: np.log10(x + 1))


	# date related features
	df['release_date'] = pd.to_datetime(df['release_date'])
	df['release_year'] = df['release_date'].dt.year
	df['release_month'] = df['release_date'].dt.month
	df['passed_years'] = date.today().year - df['release_year']
	df['release_season'] = pd.cut(df['release_month'], bins=[0, 3, 6, 9, 12],
								  labels=["Winter", "Spring", "Summer", "Autumn"]).astype('category')

	labels = df[target]
	df.drop(['release_date', 'release_month',target,'id'], axis=1,inplace=True)

	cat_vars = ['homepage', 'original_language', 'video', 'production_company_country', 'director_id',
				'producer_id', 'collection_id', 'production_company_id', 'genres', 'top_3_actors']\
			   +[col for col in df.columns if 'genre_' in col]
	numerical_vars = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count','avg_salary']
	features = shrink_memory_consumption(df,cat_vars,numerical_vars)
	if isinstance(features_imp,str):
		features_imp = np.zeros((features.shape[1],))

	features,valid_features,labels,valid_labels = train_test_split(features,labels,test_size=0.15,random_state=314)

	#
	# model = CatBoostRegressor()
	# parameters = {'depth': [8, 10, 12],
	# 				'learning_rate': [0.01, 0.05, 0.005],
	# 				'iterations': [ 150, 100, 200],
	# 				'l2_leaf_reg' : [0.1],
	# 				'loss_function' : ['RMSE'],
	# 				'eval_metric' : ['RMSE'],
	# 				'cat_features' : [features.select_dtypes('category').columns],
	# 				'random_strength' : [0.001],
	# 				'bootstrap_type' : ['Bayesian'],
	# 				'bagging_temperature' : [1],
	# 				'leaf_estimation_method' : ['Newton'],
	# 				'leaf_estimation_iterations' : [2],
	# 				'boosting_type' : ['Ordered'],
	# 				'feature_border_type' : ['Median'],
	# 				'random_seed' : [7]
	# 			  }
	# grid = GridSearchCV(estimator=model, param_grid=parameters, cv=7, n_jobs=-1)

	fit_params = {"early_stopping_rounds": 30,
				  "eval_metric": 'rmse',
				  "eval_set": [(valid_features, valid_labels)],
				  'eval_names': ['valid'],
				  # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
				  'categorical_feature': 'auto'}

	from scipy.stats import randint as sp_randint
	from scipy.stats import uniform as sp_uniform
	param_test = {'num_leaves': sp_randint(6, 50),
				  'min_child_samples': sp_randint(100, 500),
				  'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
				  'subsample': sp_uniform(loc=0.2, scale=0.8),
				  'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
				  'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10],
				  'reg_lambda': [0, 1e-1, 1, 5, 10, 20]}

	clf = lgb.LGBMRegressor(max_depth=-1, random_state=314, silent=True,objective='regression', n_jobs=4, n_estimators=1000)
	grid = RandomizedSearchCV(
		estimator=clf, param_distributions=param_test,
		scoring='neg_root_mean_squared_error',
		cv=5,
		refit=True,
		random_state=314,
		verbose=True)

	grid.fit(features,labels,**fit_params)

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
	predictors = [x for x in features.columns]

	plot_df = pd.DataFrame([[i[0], i[1]] for i in zip(features_imp, predictors)], columns=['importance', 'feature'])
	plot_feature_importances(plot_df)

def experiment_3(df_orig,target,model_name,plot=False):
	# Create the kfold object
	number_of_folds = 7
	k_fold = KFold(n_splits=number_of_folds, shuffle=True, random_state=50)
	valid_scores = []
	train_scores = []
	df = deepcopy(df_orig)
	predictors = []
	features_imp = 'empty'

	for column in list(df.select_dtypes('number').columns):
		if abs(df[column].skew()) > 1.0:
			df[column] = df[column].apply(lambda x: np.log10(x + 1))

	# Iterate through each fold
	for train_indices, valid_indices in k_fold.split(df):
		train_features, train_labels = df.loc[train_indices], df[target].loc[train_indices]
		valid_features, valid_labels = df.loc[valid_indices], df[target].loc[valid_indices]

		# build train features
		df_built = build_features(train_features)
		df_built = enrich_features_with(train_features,df_built,['genres','top_3_actors'])
		train_features, cols_to_remove = delete_highly_target_correlated_features(df_built, target)
		train_features.drop([target],axis=1,inplace=True)
		print(train_features.shape)

		# build test features
		df_built = build_features(valid_features)
		valid_features = enrich_features_with(valid_features,df_built,['genres','top_3_actors'])
		valid_features = valid_features.align(train_features,join='right',axis=1)[0]

		group_by_variables = ['release_season', 'collection_id', 'production_company_id', 'director_id', 'producer_id',
							  'genres','top_3_actors']
		for idx in valid_features.index:
			for var in group_by_variables:
				relevant_columns = [_ for _, col in enumerate(train_features.columns) if str(col).startswith(var) and col!=var]
				if valid_features.loc[idx,var] in list(train_features[var]):
					idx2 = train_features.index[train_features[var]==valid_features.loc[idx,var]][0]
					valid_features.iloc[idx,relevant_columns] = train_features.iloc[idx2,relevant_columns]
				# else:
				# 	valid_features.iloc[idx, relevant_columns] = train_features[train_features.columns[relevant_columns]].mean()

		columns_to_impute = valid_features.isna().sum().index[valid_features.isna().sum() > 0]
		for col in columns_to_impute:
			valid_features[col] = valid_features[col].fillna(train_features[col].mean())

		train_features.drop(['id','release_date'], axis=1, inplace=True)
		valid_features.drop(['id','release_date'], axis=1, inplace=True)

		#Create the model
		if model_name=='LGBM':
			model, valid_score, train_score = LGMB_model(train_features,train_labels,valid_features,valid_labels)
		elif model_name=='CATBOOST':
			model, valid_score, train_score = CatBoost_model(train_features, train_labels, valid_features, valid_labels)

		valid_scores.append(valid_score)
		train_scores.append(train_score)

		features_imp = model.feature_importances_
		predictors = [x for x in train_features.columns]

		if plot:
			plot_df = pd.DataFrame([[i[0], i[1]] for i in zip(features_imp, predictors)], columns=['importance', 'feature'])
			plot_feature_importances(plot_df)

	print('Avg train loss is {} million dollars'.format((10 ** np.mean(train_scores))))
	print('Avg valid loss is {} million dollars'.format((10 ** np.mean(valid_scores))))

	return (10 ** np.mean(train_scores)), (10 ** np.mean(valid_scores))


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
	ax.barh(list(reversed(list(df.index[:15]))),
			df['importance_normalized'].head(15),
			align='center', edgecolor='k')

	# Set the yticks and labels
	ax.set_yticks(list(reversed(list(df.index[:15]))))
	ax.set_yticklabels(df['feature'].head(15))

	# Plot labeling
	plt.xlabel('Normalized Importance');
	plt.title('Feature Importances')
	plt.show()

	return df

if __name__=='__main__':
	raw_data = pd.read_csv('data/train.tsv', sep="\t")
	df = preprocess(raw_data)
	# df = pd.read_csv('./data/clean_data.csv')

	# models = ['LGBM','CATBOOST']
	# final_results =[]
	# for model in models:
	# 	result = experiment_1(df,'revenue',model)
	# 	# result = experiment_3(df, 'revenue',model)
	# 	final_results.append(result)
	# print(final_results)

	experiment_2(df,'revenue')