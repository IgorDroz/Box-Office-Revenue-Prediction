from ast import literal_eval as parser
import scipy.stats as ss
from collections import Counter
import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from datetime import date
import pickle



# import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, KFold
# from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
# stop = set(stopwords.words('english'))
# import os
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
# import plotly.graph_objs as go
# import plotly.tools as tls
import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import json
import ast
import eli5
# import shap
from catboost import CatBoostRegressor
from urllib.request import urlopen
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model



####################################################################
###################### General Utils ###############################
####################################################################

def shrink_memory_consumption(data,cat_vals=None,numerical_vals=None):
	"""
	cat_vals = ['genres', 'homepage', 'original_language', 'video', 'production_company_country', 'director_id',
				'producer_id', 'top_3_actors', 'collection_id' , 'production_company_id']
	numerical_vals = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']

	df = shrink_memory_consumption(data,cat_vals,numerical_vals)
	dtypes = df.dtypes
	colnames = dtypes.index
	types = [i.name for i in dtypes.values]
	column_types = dict(zip(colnames, types))
	pickle.dump(column_types,open('./data/column_types','wb'))
	df.to_csv('./data/clean_data.csv',header=True,index=False)

	"""

	# reduce memory consumption
	if numerical_vals==None:
		numerical_vals =data.select_dtypes(include=np.number).columns.tolist()

	if cat_vals==None:
		# ToDo : complete this by thershold the unique val
		pass

	for cat in cat_vals:
		data[cat] = data[cat].astype(str)
		data[cat] = data[cat].astype('category')

	for num in numerical_vals:
		if isinstance(data[num].iloc[0],float):

			data[num] = pd.to_numeric(data[num], downcast='float')
		else:
			data[num] = pd.to_numeric(data[num], downcast='unsigned')

	return data

def delete_highly_target_correlated_features(df,target,threshold=0.9):
	corrs = df.corr()
	corrs = corrs.sort_values(target, ascending=False)
	# Empty dictionary to hold correlated variables
	above_threshold_vars = {}

	# For each column, record the variables that are above the threshold
	for col in corrs:
		above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])

	# Track columns to remove and columns already examined
	cols_to_remove = []
	cols_seen = []
	cols_to_remove_pair = []

	# Iterate through columns and correlated columns
	for key, value in above_threshold_vars.items():
		# Keep track of columns already examined
		cols_seen.append(key)
		for x in value:
			if x == key:
				continue
			else:
				# Only want to remove one in a pair
				if x not in cols_seen:
					cols_to_remove.append(x)
					cols_to_remove_pair.append(key)

	cols_to_remove = list(set(cols_to_remove))
	print('Number of columns that are removed: ', len(cols_to_remove))
	df.drop(cols_to_remove,axis=1,inplace=True)
	return df,cols_to_remove

def convert(data, to):
	converted = None
	if to == 'array':
		if isinstance(data, np.ndarray):
			converted = data
		elif isinstance(data, pd.Series):
			converted = data.values
		elif isinstance(data, list):
			converted = np.array(data)
		elif isinstance(data, pd.DataFrame):
			converted = data.as_matrix()
	elif to == 'list':
		if isinstance(data, list):
			converted = data
		elif isinstance(data, pd.Series):
			converted = data.values.tolist()
		elif isinstance(data, np.ndarray):
			converted = data.tolist()
	elif to == 'dataframe':
		if isinstance(data, pd.DataFrame):
			converted = data
		elif isinstance(data, np.ndarray):
			converted = pd.DataFrame(data)
	else:
		raise ValueError("Unknown data conversion: {}".format(to))
	if converted is None:
		raise TypeError('cannot handle data conversion of type: {} to {}'.format(type(data), to))
	else:
		return converted


def conditional_entropy(x, y):
	"""
	Calculates the conditional entropy of x given y: S(x|y)
	Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
	:param x: list / NumPy ndarray / Pandas Series
		A sequence of measurements
	:param y: list / NumPy ndarray / Pandas Series
		A sequence of measurements
	:return: float
	"""
	# entropy of x given y
	y_counter = Counter(y)
	xy_counter = Counter(list(zip(x, y)))
	total_occurrences = sum(y_counter.values())
	entropy = 0.0
	for xy in xy_counter.keys():
		p_xy = xy_counter[xy] / total_occurrences
		p_y = y_counter[xy[1]] / total_occurrences
		entropy += p_xy * math.log(p_y / p_xy)
	return entropy


def cramers_v(x, y):
	confusion_matrix = pd.crosstab(x, y)
	chi2 = ss.chi2_contingency(confusion_matrix)[0]
	n = confusion_matrix.sum().sum()
	phi2 = chi2 / n
	r, k = confusion_matrix.shape
	phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
	rcorr = r - ((r - 1) ** 2) / (n - 1)
	kcorr = k - ((k - 1) ** 2) / (n - 1)
	return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def theils_u(x, y):
	s_xy = conditional_entropy(x, y)
	x_counter = Counter(x)
	total_occurrences = sum(x_counter.values())
	p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
	s_x = ss.entropy(p_x)
	if s_x == 0:
		return 1
	else:
		return (s_x - s_xy) / s_x


def correlation_ratio(categories, measurements):
	fcat, _ = pd.factorize(categories)
	cat_num = np.max(fcat) + 1
	y_avg_array = np.zeros(cat_num)
	n_array = np.zeros(cat_num)
	for i in range(0, cat_num):
		cat_measures = measurements[np.argwhere(fcat == i).flatten()]
		n_array[i] = len(cat_measures)
		y_avg_array[i] = np.average(cat_measures)
	y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
	numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
	denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
	if numerator == 0:
		eta = 0.0
	else:
		eta = numerator / denominator
	return eta


def associations(dataset, nominal_columns=None, mark_columns=False, theil_u=False, plot=True,
				 return_results=False, **kwargs):
	"""
	Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
	continuous features using:
	 - Pearson's R for continuous-continuous cases
	 - Correlation Ratio for categorical-continuous cases
	 - Cramer's V or Theil's U for categorical-categorical cases
	:param dataset: NumPy ndarray / Pandas DataFrame
		The data-set for which the features' correlation is computed
	:param nominal_columns: string / list / NumPy ndarray
		Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
		columns are categorical, or None (default) to state none are categorical
	:param mark_columns: Boolean (default: False)
		if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
		continuous), as provided by nominal_columns
	:param theil_u: Boolean (default: False)
		In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
	:param plot: Boolean (default: True)
		If True, plot a heat-map of the correlation matrix
	:param return_results: Boolean (default: False)
		If True, the function will return a Pandas DataFrame of the computed associations
	:param kwargs:
		Arguments to be passed to used function and methods
	:return: Pandas DataFrame
		A DataFrame of the correlation/strength-of-association between all features
	"""

	dataset = convert(dataset, 'dataframe')
	columns = dataset.columns
	if nominal_columns is None:
		nominal_columns = list()
	elif nominal_columns == 'all':
		nominal_columns = columns
	corr = pd.DataFrame(index=columns, columns=columns)
	for i in range(0, len(columns)):
		for j in range(i, len(columns)):
			if i == j:
				corr[columns[i]][columns[j]] = 1.0
			else:
				if columns[i] in nominal_columns:
					if columns[j] in nominal_columns:
						if theil_u:
							corr[columns[j]][columns[i]] = theils_u(dataset[columns[i]], dataset[columns[j]])
							corr[columns[i]][columns[j]] = theils_u(dataset[columns[j]], dataset[columns[i]])
						else:
							cell = cramers_v(dataset[columns[i]], dataset[columns[j]])
							corr[columns[i]][columns[j]] = cell
							corr[columns[j]][columns[i]] = cell
					else:
						cell = correlation_ratio(dataset[columns[i]], dataset[columns[j]])
						corr[columns[i]][columns[j]] = cell
						corr[columns[j]][columns[i]] = cell
				else:
					if columns[j] in nominal_columns:
						cell = correlation_ratio(dataset[columns[j]], dataset[columns[i]])
						corr[columns[i]][columns[j]] = cell
						corr[columns[j]][columns[i]] = cell
					else:
						cell, _ = ss.pearsonr(dataset[columns[i]], dataset[columns[j]])
						corr[columns[i]][columns[j]] = cell
						corr[columns[j]][columns[i]] = cell
	corr.fillna(value=np.nan, inplace=True)
	if mark_columns:
		marked_columns = ['{} (nom)'.format(col) if col in nominal_columns else '{} (con)'.format(col) for col in
						  columns]
		corr.columns = marked_columns
		corr.index = marked_columns
	if plot:
		plt.figure(figsize=(20, 20))  # kwargs.get('figsize',None))
		sns.heatmap(corr, annot=kwargs.get('annot', True), fmt=kwargs.get('fmt', '.2f'), cmap='coolwarm')
		plt.show()
	if return_results:
		return corr


def agg_numeric(df, group_var, df_name):
	"""Aggregates the numeric values in a dataframe. This can
	be used to create features for each instance of the grouping variable.

	Parameters
	--------
		df (dataframe):
			the dataframe to calculate the statistics on
		group_var (string):
			the variable by which to group df
		df_name (string):
			the variable used to rename the columns

	Return
	--------
		agg (dataframe):
			a dataframe with the statistics aggregated for
			all numeric columns. Each instance of the grouping variable will have
			the statistics (mean, min, max, sum; currently supported) calculated.
			The columns are also renamed to keep track of features created.

	"""

	group_ids = df[group_var]
	numeric_df = df.select_dtypes('number')
	numeric_df[group_var] = group_ids

	# Group by the specified variable and calculate the statistics
	agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
	# Need to create new column names
	columns = [group_var]

	# Iterate through the variables names
	for var in agg.columns.levels[0]:
		# Skip the grouping variable
		if var != group_var:
			# Iterate through the stat names
			for stat in agg.columns.levels[1][:-1]:
				# Make a new column name for the variable and stat
				columns.append('%s_%s_%s' % (df_name, var, stat))

	agg.columns = columns
	agg = shrink_memory_consumption(agg,[],columns[1:])
	return agg


def count_categorical(df, group_var, df_name,exclude = None):
	"""Computes counts and normalized counts for each observation
	of `group_var` of each unique category in every categorical variable

	Parameters
	--------
	df : dataframe
		The dataframe to calculate the value counts for.

	group_var : string
		The variable by which to group the dataframe. For each unique
		value of this variable, the final dataframe will have one row

	df_name : string
		Variable added to the front of column names to keep track of columns


	Return
	--------
	categorical : dataframe
		A dataframe with counts and normalized counts of each unique category in every categorical variable
		with one row for every unique value of the `group_var`.

	"""

	# Select the categorical columns

	if exclude:
		try:
			exclude.remove(group_var)
		except:
			pass
		df.drop(exclude, axis=1, inplace=True)
	categorical = pd.get_dummies(df.select_dtypes('category').drop([group_var],axis=1))
	filter_col = categorical.columns

	# Make sure to put the identifying id on the column
	categorical[group_var] = df[group_var]

	# Groupby the group var and calculate the sum and mean
	categorical = categorical.groupby(group_var)[filter_col].agg(['sum', 'mean'])

	column_names = []

	# Iterate through the columns in level 0
	for var in categorical.columns.levels[0]:
		# Iterate through the stats in level 1
		for stat in ['count', 'count_norm']:
			# Make a new column name
			column_names.append('%s_%s_%s' % (df_name, var, stat))

	categorical.columns = column_names
	categorical = shrink_memory_consumption(categorical,[],column_names)

	return categorical

####################################################################
###################### Specific Utils ##############################
####################################################################

def preprocess(data,ohe=False, save=True):
	data.drop(['imdb_id','poster_path','original_title','production_countries',
			   'title','status','spoken_languages','Keywords','overview'],axis=1,inplace=True)

	data['got_img'] = np.where(data['backdrop_path'].isna(),0,1)
	data['tagline'] = np.where(data['tagline'].isna(), 0, 1)
	data['homepage'] = np.where(data['homepage'].isna(),0,1)
	data['video'] = np.where(data['video']==True,1,0)
	data['budget'] = np.where(data['budget'] == 0, 1, data['budget'])
	data['vote_count'] = np.where(data['vote_count'] == 0, 1, data['vote_count'])
	data['runtime'] = data['runtime'].fillna(np.median(data['runtime'].dropna()))
	data.drop(['backdrop_path'], axis=1, inplace=True)

	# extract collection id
	data['belongs_to_collection'] = data['belongs_to_collection'].fillna('{}')
	uniqueAttr = set()
	data['collection_id'] = 'No Collection'
	for idx,cell in enumerate(data['belongs_to_collection']):
		cell = parser(cell)
		uniqueAttr = uniqueAttr.union(set(cell.keys()))
		for key,val in cell.items():
			if key == 'id':
				data['collection_id'].loc[idx] = str(val)
	data.drop(['belongs_to_collection'],axis=1,inplace=True)


	# extract genres (There are 19 of them)
	data['genres'] = np.where(data['genres']=='[]','[{\'name\' : \'Other\'}]',data['genres'])
	for idx,cell in enumerate(data['genres']):
		cell = parser(cell)
		combined_genre =[]
		for genre in cell:
			combined_genre.append(genre['name'])
			if ohe:
				if 'genre_' + genre['name'] not in data.columns:
					data['genre_' + genre['name']] = 0
				data['genre_' + genre['name']].loc[idx] = 1

		data['genres'].loc[idx] = combined_genre
	# data.drop(['genres'],axis=1,inplace=True)

	# extract production company details
	data['production_company_id'] = 'No id'
	data['production_company_country'] = 'No country'
	data['production_companies_involved'] = 1
	for idx,cell in enumerate(data['production_companies']):
		cell = parser(cell)
		data['production_companies_involved'].loc[idx] = len(cell)
		for element in cell:
			for key,val in element.items():
				if key=='id':
					data['production_company_id'].loc[idx] = str(val)
				elif key=='origin_country':
					data['production_company_country'].loc[idx] = val if val!='' else 'No country'
				break
	data.drop(['production_companies'],axis=1,inplace=True)

	# extract top 3 actors
	data['top_3_actors'] = '[\'Other\']'
	data['actors_involved'] = 1
	data['cast'] = np.where(data['cast'] == '[]', '[{\'id\' : \'Other\'}]', data['cast'])
	for idx,cell in enumerate(data['cast']):
		cell = parser(cell)
		data['actors_involved'].loc[idx] = len(cell)
		combined_actors = []
		counter = 0
		for element in cell:
			if counter==3:
				break
			combined_actors.append(str(element['id']))
			counter+=1
		data['top_3_actors'].loc[idx] = combined_actors
	data.drop(['cast'],axis=1,inplace=True)

	# extract director and producer
	data['director_id'] = 'No director'
	data['producer_id'] = 'No producer'
	data['crew_involved'] = 1
	for idx,cell in enumerate(data['crew']):
		cell = parser(cell)
		# data['crew_involved'].loc[idx] = len(cell)
		crew_set =set()
		producer=0
		director=0
		for element in cell:
			if producer and director:
				if director<1 and 'director' in element['job'].lower():
					data['director_id'].loc[idx] = str(element['id'])
					director+=1

				if producer<1 and 'producer' in element['job'].lower():
					data['producer_id'].loc[idx] = str(element['id'])
					producer+=1
			crew_set.add(str(element['id']))
		data['crew_involved'].loc[idx] = len(crew_set)

	data.drop(['crew'],axis=1,inplace=True)

	data['avg_salary']=data['budget']/(data['crew_involved']+data['actors_involved'])

	if save:
		data.to_csv('./data/clean_data.csv',header=True,index=False)

	return data


# creating features based on dates
def process_date(df):
	date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']
	for part in date_parts:
		part_col = 'release_date' + "_" + part
		df[part_col] = getattr(df['release_date'].dt, part).astype(int)

	return df

def preprocess_shir(train, test, save=True):

	print('########### strating preprocess_shir ###########')

	#add log revenue and budget
	train['log_revenue'] = np.log1p(train['revenue'])
	train['log_budget'] = np.log1p(train['budget'])

	test['log_budget'] = np.log1p(test['budget'])
	test['log_revenue'] = np.log1p(test['revenue'])

	#homepage
	train['has_homepage'] = 0
	train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
	test['has_homepage'] = 0
	test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1

	#release date
	train['release_date'] = pd.to_datetime(train['release_date'])
	test['release_date'] = pd.to_datetime(test['release_date'])
	train = process_date(train)
	test = process_date(test)

	#collections
	#does the film belongs to colection or not
	train['belongs_to_collection'] = train['belongs_to_collection'].fillna('{}')
	train['collection_name'] = train['belongs_to_collection'].apply(lambda x: parser(x)['name'] if ((x != {}) and ('name' in parser(x).keys())) else 0)
	train['has_collection'] = train['collection_name'].apply(lambda x: 1 if x != 0 else 0)

	test['belongs_to_collection'] = test['belongs_to_collection'].fillna('{}')
	test['collection_name'] = test['belongs_to_collection'].apply(lambda x: parser(x)['name'] if ((x != {}) and ('name' in parser(x).keys())) else 0)
	test['has_collection'] = test['collection_name'].apply(lambda x: 1 if x != 0 else 0)

	train = train.drop(['belongs_to_collection'], axis=1)
	test = test.drop(['belongs_to_collection'], axis=1)

	#genres
	#create separate columns for each genre (19 geners in training)
	list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)
	set_of_genres = set([m for m in Counter([i for j in list_of_genres for i in j])])

	train['genres'] = train['genres'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	train['num_of_geners'] = train['genres'].apply(lambda x: len(x))
	for g in set_of_genres:
		train['genre_' + g] = train['genres'].apply(lambda x: 1 if g in x else 0)

	test['genres'] = test['genres'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	test['num_of_geners'] = test['genres'].apply(lambda x: len(x))
	for g in set_of_genres:
		test['genre_' + g] = test['genres'].apply(lambda x: 1 if g in x else 0)

	#production companies
	list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)

	train['all_production_companies'] = train['production_companies'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	train['num_companies'] = train['all_production_companies'].apply(lambda x: len(x))
	top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]
	for g in top_companies:
		train['production_company_' + g] = train['all_production_companies'].apply(lambda x: 1 if g in x else 0)

	test['all_production_companies'] = test['production_companies'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	test['num_companies'] = test['all_production_companies'].apply(lambda x: len(x))
	for g in top_companies:
		test['production_company_' + g] = test['all_production_companies'].apply(lambda x: 1 if g in x else 0)

	# train = train.drop(['production_companies', 'all_production_companies'], axis=1)
	# test = test.drop(['production_companies', 'all_production_companies'], axis=1)
	train = train.drop(['production_companies'], axis=1)
	test = test.drop(['production_companies'], axis=1)

	#production countries
	list_of_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)

	train['all_countries'] = train['production_countries'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	train['num_countries'] = train['all_countries'].apply(lambda x: len(x))
	top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]
	for g in top_countries:
		train['production_country_' + g] = train['all_countries'].apply(lambda x: 1 if g in x else 0)

	test['all_countries'] = test['production_countries'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	test['num_countries'] = test['all_countries'].apply(lambda x: len(x))
	for g in top_countries:
		test['production_country_' + g] = test['all_countries'].apply(lambda x: 1 if g in x else 0)

	# train = train.drop(['production_countries', 'all_countries'], axis=1)
	# test = test.drop(['production_countries', 'all_countries'], axis=1)
	train = train.drop(['production_countries'], axis=1)
	test = test.drop(['production_countries'], axis=1)

	#spoken languages
	list_of_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)

	train['all_languages'] = train['spoken_languages'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	train['num_languages'] = train['all_languages'].apply(lambda x: len(x))
	top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]
	for g in top_languages:
		train['language_' + g] = train['all_languages'].apply(lambda x: 1 if g in x else 0)

	test['all_languages'] = test['spoken_languages'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	test['num_languages'] = test['all_languages'].apply(lambda x: len(x))
	for g in top_languages:
		test['language_' + g] = test['all_languages'].apply(lambda x: 1 if g in x else 0)

	# train = train.drop(['spoken_languages', 'all_languages'], axis=1)
	# test = test.drop(['spoken_languages', 'all_languages'], axis=1)
	train = train.drop(['spoken_languages'], axis=1)
	test = test.drop(['spoken_languages'], axis=1)

	#keywords
	list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)

	train['all_Keywords'] = train['Keywords'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	train['num_Keywords'] = train['all_Keywords'].apply(lambda x: len(x))
	top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]
	for g in top_keywords:
		train['keyword_' + g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)

	test['all_Keywords'] = test['Keywords'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	test['num_Keywords'] = test['all_Keywords'].apply(lambda x: len(x))
	for g in top_keywords:
		test['keyword_' + g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)

	# train = train.drop(['Keywords', 'all_Keywords'], axis=1)
	# test = test.drop(['Keywords', 'all_Keywords'], axis=1)
	train = train.drop(['Keywords'], axis=1)
	test = test.drop(['Keywords'], axis=1)

	#cast
	list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)
	list_of_cast_genders = list(train['cast'].apply(lambda x: [i['gender'] for i in parser(x)] if x != {} else []).values)
	list_of_cast_characters = list(train['cast'].apply(lambda x: [i['character'] for i in parser(x)] if x != {} else []).values)

	train['all_cast'] = train['cast'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	train['num_cast'] = train['all_cast'].apply(lambda x: len(x))
	top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]

	for g in top_cast_names:
		train['cast_name_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)
	train['genders_0_cast'] = train['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 0]))
	train['genders_1_cast'] = train['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 1]))
	train['genders_2_cast'] = train['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 2]))
	top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]
	for g in top_cast_characters:
		train['cast_character_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)

	test['all_cast'] = test['cast'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	test['num_cast'] = test['all_cast'].apply(lambda x: len(x))
	for g in top_cast_names:
		test['cast_name_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)
	test['genders_0_cast'] = test['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 0]))
	test['genders_1_cast'] = test['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 1]))
	test['genders_2_cast'] = test['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 2]))
	for g in top_cast_characters:
		test['cast_character_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)

	# train = train.drop(['cast','all_cast'], axis=1)
	# test = test.drop(['cast','all_cast'], axis=1)
	train = train.drop(['cast'], axis=1)
	test = test.drop(['cast'], axis=1)

	#crew
	list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)
	list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in parser(x)] if x != {} else []).values)
	list_of_crew_genders = list(train['crew'].apply(lambda x: [i['gender'] for i in parser(x)] if x != {} else []).values)
	list_of_crew_departments  = list(train['crew'].apply(lambda x: [i['department'] for i in parser(x)] if x != {} else []).values)

	train['all_crew'] = train['crew'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	train['num_crew'] = train['all_crew'].apply(lambda x: len(x))
	top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
	for g in top_crew_names:
		train['crew_name_' + g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)
	train['genders_0_crew'] = train['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 0]))
	train['genders_1_crew'] = train['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 1]))
	train['genders_2_crew'] = train['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 2]))
	top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]
	for j in top_crew_jobs:
		train['jobs_' + j] = train['crew'].apply(lambda x: sum([1 for i in parser(x) if i['job'] == j]))
	top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]
	for j in top_crew_departments:
		train['departments_' + j] = train['crew'].apply(lambda x: sum([1 for i in parser(x) if i['department'] == j]))

	test['all_crew'] = test['crew'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
	test['num_crew'] = test['all_crew'].apply(lambda x: len(x))
	for g in top_crew_names:
		test['crew_name_' + g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)
	test['genders_0_crew'] = test['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 0]))
	test['genders_1_crew'] = test['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 1]))
	test['genders_2_crew'] = test['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 2]))
	for j in top_crew_jobs:
		test['jobs_' + j] = test['crew'].apply(lambda x: sum([1 for i in parser(x) if i['job'] == j]))
	for j in top_crew_departments:
		test['departments_' + j] = test['crew'].apply(lambda x: sum([1 for i in parser(x) if i['department'] == j]))

	# train = train.drop(['crew','all_crew'], axis=1)
	# test = test.drop(['crew','all_crew'], axis=1)
	train = train.drop(['crew'], axis=1)
	test = test.drop(['crew'], axis=1)

	if save:
		train.to_csv('./data/clean_train.csv',header=True,index=False)
		test.to_csv('./data/clean_test.csv',header=True,index=False)

	print('########### finished preprocess_shir ###########')


def train_model(X, X_test, y, folds, params=None, model_type='lgb', plot_feature_importance=False, model=None):
	n_fold = 10
	# folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

	oof = np.zeros(X.shape[0])
	prediction = np.zeros(X_test.shape[0])
	scores = []
	feature_importance = pd.DataFrame()
	for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
		print('Fold', fold_n, 'started at', time.ctime())
		if model_type == 'sklearn':
			X_train, X_valid = X[train_index], X[valid_index]
		else:
			X_train, X_valid = X.values[train_index], X.values[valid_index]
		y_train, y_valid = y[train_index], y[valid_index]

		if model_type == 'lgb':
			model = lgb.LGBMRegressor(**params, n_estimators=20000, nthread=4, n_jobs=-1)
			model.fit(X_train, y_train,
					  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
					  verbose=1000, early_stopping_rounds=200)
			y_pred_valid = model.predict(X_valid)
			y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

		if model_type == 'xgb':
			train_data = xgb.DMatrix(data=X_train, label=y_train)
			valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

			watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
			model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
							  verbose_eval=500, params=params)
			y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
			y_pred = model.predict(xgb.DMatrix(X_test.values), ntree_limit=model.best_ntree_limit)

		if model_type == 'sklearn':
			model = model
			model.fit(X_train, y_train)
			y_pred_valid = model.predict(X_valid).reshape(-1, )
			score = mean_squared_error(y_valid, y_pred_valid)
			print("mean_squared_error: ", score)

			y_pred = model.predict(X_test)

		if model_type == 'cat':
			model = CatBoostRegressor(iterations=20000, eval_metric='RMSE', **params)
			model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
					  verbose=False)

			y_pred_valid = model.predict(X_valid)
			y_pred = model.predict(X_test)

		oof[valid_index] = y_pred_valid.reshape(-1, )
		scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)

		prediction += y_pred

		if model_type == 'lgb':
			# feature importance
			fold_importance = pd.DataFrame()
			fold_importance["feature"] = X.columns
			fold_importance["importance"] = model.feature_importances_
			fold_importance["fold"] = fold_n + 1
			feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

	prediction /= n_fold
	print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

	if model_type == 'lgb':
		feature_importance["importance"] /= n_fold
		if plot_feature_importance:
			cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
				by="importance", ascending=False)[:50].index

			best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

			plt.figure(figsize=(16, 12));
			sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
			plt.title('LGB Features (avg over folds)');

			return oof, prediction, feature_importance
		return oof, prediction

	else:
		return oof, prediction


def integration(train,test):
	#add productions information
	production_df = pd.read_csv("./data/production.csv",encoding="ISO-8859-1")

	production_df['total_revenue'] = production_df.total_revenue.str.replace(r'\D','').astype(np.int64)
	production_df['avg_revenue_for_movie'] = production_df.apply(lambda x: x.total_revenue / x.number_of_movies, axis=1)

	train['production_number_of_movies'] = 0
	train['production_avg_rev_for_movie'] = 0
	train['production_total_rev'] = 0

	#collections
	collectiond_df = pd.read_csv("./data/collections.csv",encoding="ISO-8859-1")
	collectiond_df['revnue'] = collectiond_df.revnue.str.replace(r'\D','').astype(np.int64)
	collectiond_df['avg_revenue_for_movie'] = collectiond_df.apply(lambda x: x.revnue / x.number_of_movies, axis=1)
	collectiond_df['collection'] = collectiond_df['collection'].str.lower()

	train['collection_name'] = train['collection_name'].str.lower()
	train['collection_name'] = train['collection_name'].str.replace(r' collection','')

	train['collection_number_of_movies'] = 0
	train['collection_avg_rev_for_movie'] = 0
	train['collection_total_rev'] = 0

	#years
	years_df = pd.read_csv("./data/years.csv",encoding="ISO-8859-1")
	years_df = years_df.fillna('0')
	years_df['number_of_movies'] = years_df.number_of_movies.str.replace(r'\D','').astype(np.int64)
	years_df['total_revenue'] = years_df.total_revenue.str.replace(r'\D','').astype(np.int64)
	years_df['avg_revenue_for_movie'] = years_df.apply(lambda x: x.total_revenue / x.number_of_movies, axis=1)


	train['year_number_of_movies'] = 0
	train['year_avg_rev_for_movie'] = 0
	train['year_total_rev'] = 0

	for index, row in train.iterrows():
		#production
		production_companies = parser(row['all_production_companies'])
		rows_prod = production_df.loc[production_df['production'].isin(production_companies)]
		number_of_movies = list(rows_prod['number_of_movies'])
		avg_rev_for_movie = list(rows_prod['avg_revenue_for_movie'])
		production_total_rev = list(rows_prod['total_revenue'])
		if number_of_movies:
			train['production_number_of_movies'][index] = max(number_of_movies)
			train['production_avg_rev_for_movie'][index] = max(avg_rev_for_movie)
			train['production_total_rev'][index] = max(production_total_rev)

		#collections
		collection = row['collection_name']
		rows_collection = collectiond_df.loc[collectiond_df['collection'] == collection]
		if not rows_collection.empty:
			train['collection_number_of_movies'][index] = list(rows_collection['number_of_movies'])[0]
			train['collection_avg_rev_for_movie'][index] = list(rows_collection['avg_revenue_for_movie'])[0]
			train['collection_total_rev'][index] = list(rows_collection['revnue'])[0]

		#years
		year = row['release_date_year']
		rows_year = years_df.loc[years_df['year'] == year]
		if not rows_year.empty:
			rows_year = rows_year.fillna(0)
			train['year_number_of_movies'][index] = list(rows_year['number_of_movies'])[0]
			train['year_avg_rev_for_movie'][index] = list(rows_year['avg_revenue_for_movie'])[0]
			train['year_total_rev'][index] = list(rows_year['total_revenue'])[0]

	print()


def new_features(df):
	df['budget_to_popularity'] = df['budget'] / df['popularity']
	df['budget_to_runtime'] = df['budget'] / df['runtime']

	# some features from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3
	df['_budget_year_ratio'] = df['budget'] / (df['release_date_year'] * df['release_date_year'])
	df['_releaseYear_popularity_ratio'] = df['release_date_year'] / df['popularity']
	df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_date_year']

	df['runtime_to_mean_year'] = df['runtime'] / df.groupby("release_date_year")["runtime"].transform('mean')
	df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_date_year")["popularity"].transform('mean')
	df['budget_to_mean_year'] = df['budget'] / df.groupby("release_date_year")["budget"].transform('mean')

	return df


def preparing_data_to_expirement(train, test):
	train = train.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue','backdrop_path'], axis=1)
	test = test.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status','log_revenue','backdrop_path'], axis=1)

	for col in train.columns:
		if train[col].nunique() == 1:
			print(col)
			train = train.drop([col], axis=1)
			test = test.drop([col], axis=1)

	for col in ['original_language', 'collection_name', 'genres']:
		le = LabelEncoder()
		le.fit(list(train[col].fillna('')) + list(test[col].fillna('')))
		train[col] = le.transform(train[col].fillna('').astype(str))
		test[col] = le.transform(test[col].fillna('').astype(str))

	train_texts = train[['title', 'tagline', 'overview', 'original_title']]
	test_texts = test[['title', 'tagline', 'overview', 'original_title']]

	for col in ['title', 'tagline', 'overview', 'original_title']:
		train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))
		train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))
		train = train.drop(col, axis=1)
		test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))
		test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))
		test = test.drop(col, axis=1)

	X = train.drop(['id', 'revenue'], axis=1)
	y = np.log1p(train['revenue'])
	X_test = test.drop(['id','revenue'], axis=1)

	n_fold = 10
	folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

	for col in train_texts.columns:
		vectorizer = TfidfVectorizer(
			sublinear_tf=True,
			analyzer='word',
			token_pattern=r'\w{1,}',
			ngram_range=(1, 2),
			min_df=10
		)
		vectorizer.fit(list(train_texts[col].fillna('')) + list(test_texts[col].fillna('')))
		train_col_text = vectorizer.transform(train_texts[col].fillna(''))
		test_col_text = vectorizer.transform(test_texts[col].fillna(''))
		model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)
		oof_text, prediction_text = train_model(train_col_text, test_col_text, y, folds=folds, params=None, model_type='sklearn',
												model=model)

		X[col + '_oof'] = oof_text
		X_test[col + '_oof'] = prediction_text

	X = new_features(X)
	X_test = new_features(X_test)


	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

	n_fold = 10
	folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

	#building & blending

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
	oof_lgb, prediction_lgb, _ = train_model(X, X_test, y, params=params, folds=folds, model_type='lgb',
											 plot_feature_importance=True)

	results_df = test[['id', 'revenue', 'runtime']]
	results_df['revenue'] = np.expm1(prediction_lgb)
	results_df.to_csv("lgb.csv", index=False)


	xgb_params = {'eta': 0.01,
				  'objective': 'reg:linear',
				  'max_depth': 7,
				  'subsample': 0.8,
				  'colsample_bytree': 0.8,
				  'eval_metric': 'rmse',
				  'seed': 11,
				  'silent': True}
	oof_xgb, prediction_xgb = train_model(X, X_test, y, params=xgb_params, folds=folds, model_type='xgb',
										  plot_feature_importance=False)

	results_df['revenue'] = np.expm1(prediction_xgb)
	results_df.to_csv("xgb.csv", index=False)

	cat_params = {'learning_rate': 0.002,
				  'depth': 5,
				  'l2_leaf_reg': 10,
				  # 'bootstrap_type': 'Bernoulli',
				  'colsample_bylevel': 0.8,
				  'bagging_temperature': 0.2,
				  # 'metric_period': 500,
				  'od_type': 'Iter',
				  'od_wait': 100,
				  'random_seed': 11,
				  'allow_writing_files': False}
	oof_cat, prediction_cat = train_model(X, X_test, y, params=cat_params, folds=folds, model_type='cat')

	results_df['revenue'] = np.expm1(prediction_cat)
	results_df.to_csv("cat.csv", index=False)

	params = {'num_leaves': 30,
			  'min_data_in_leaf': 20,
			  'objective': 'regression',
			  'max_depth': 5,
			  'learning_rate': 0.01,
			  "boosting": "gbdt",
			  "feature_fraction": 0.9,
			  "bagging_freq": 1,
			  "bagging_fraction": 0.9,
			  "bagging_seed": 11,
			  "metric": 'rmse',
			  "lambda_l1": 0.2,
			  "verbosity": -1}
	oof_lgb_1, prediction_lgb_1 = train_model(X, X_test, y, params=params, folds=folds, model_type='lgb',
											  plot_feature_importance=False)

	results_df['revenue'] = np.expm1(prediction_lgb_1)
	results_df.to_csv("lgb1.csv", index=False)

	params = {'num_leaves': 30,
			  'min_data_in_leaf': 20,
			  'objective': 'regression',
			  'max_depth': 7,
			  'learning_rate': 0.02,
			  "boosting": "gbdt",
			  "feature_fraction": 0.7,
			  "bagging_freq": 5,
			  "bagging_fraction": 0.7,
			  "bagging_seed": 11,
			  "metric": 'rmse',
			  "lambda_l1": 0.2,
			  "verbosity": -1}
	oof_lgb_2, prediction_lgb_2 = train_model(X, X_test, y, params=params, folds=folds, model_type='lgb',
											  plot_feature_importance=False)


	results_df['revenue'] = np.expm1(prediction_lgb_2)
	results_df.to_csv("lgb_2.csv", index=False)

	train_stack = np.vstack([oof_lgb, oof_xgb, oof_cat, oof_lgb_1, oof_lgb_2]).transpose()
	train_stack = pd.DataFrame(train_stack, columns=['lgb', 'xgb', 'cat', 'lgb_1', 'lgb_2'])
	test_stack = np.vstack(
		[prediction_lgb, prediction_xgb, prediction_cat, prediction_lgb_1, prediction_lgb_2]).transpose()
	test_stack = pd.DataFrame(test_stack, columns=['lgb', 'xgb', 'cat', 'lgb_1', 'lgb_2'])

	params = {'num_leaves': 8,
			  'min_data_in_leaf': 20,
			  'objective': 'regression',
			  'max_depth': 3,
			  'learning_rate': 0.01,
			  "boosting": "gbdt",
			  "bagging_seed": 11,
			  "metric": 'rmse',
			  "lambda_l1": 0.2,
			  "verbosity": -1}
	oof_lgb_stack, prediction_lgb_stack, _ = train_model(train_stack, test_stack, y, params=params, folds=folds, model_type='lgb',
														 plot_feature_importance=True)

	model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)
	oof_rcv_stack, prediction_rcv_stack = train_model(train_stack.values, test_stack.values, y, params=None, folds=folds,
													  model_type='sklearn', model=model)


	results_df['revenue'] = prediction_lgb_stack
	results_df.to_csv("stack_lgb.csv", index=False)
	results_df['revenue'] = prediction_rcv_stack
	results_df.to_csv("stack_rcv.csv", index=False)

def enrich_dataset_by_dividing_x(data,x):
	return data.explode(x).reset_index(drop=True)

def build_statistics_features_on_numerical_vars(df,group_by_variables,original_features):
	for variable in group_by_variables:
		statistics = agg_numeric(df[original_features],variable,variable)
		df = df.merge(statistics, on = variable, how = 'left')
	return df

def build_statistics_features_on_categorical_vars(df,group_by_variables,original_features,exclude):
	for variable in group_by_variables:
		statistics = count_categorical(df[original_features],variable,variable,exclude[:])
		df = df.merge(statistics, on = variable, how = 'left')

	return df

def build_statistics_features(df,group_by_variables,original_features,exclude=None):
	df = build_statistics_features_on_numerical_vars(df, group_by_variables, original_features)
	return build_statistics_features_on_categorical_vars(df, group_by_variables, original_features ,exclude)
	# return df

def build_features(df,group_by_x=None):
	cat_vars = ['homepage', 'original_language', 'video', 'production_company_country', 'director_id', 'tagline',
				'got_img', 'producer_id', 'collection_id', 'production_company_id', 'genres', 'top_3_actors']
	numerical_vars = ['budget', 'popularity', 'runtime','revenue', 'vote_average', 'vote_count', 'crew_involved',
					  'actors_involved', 'production_companies_involved','avg_salary']

	df = shrink_memory_consumption(df,cat_vars,numerical_vars)

	# date related features
	df['release_date'] = pd.to_datetime(df['release_date'])
	df['release_year'] = df['release_date'].dt.year
	df['release_month'] = df['release_date'].dt.month
	df['passed_years'] = date.today().year - df['release_year']
	df['release_season'] = pd.cut(df['release_month'], bins=[0, 3, 6, 9, 12], labels=["Winter", "Spring", "Summer","Autumn"]).astype('category')
	#todo: mauby drop 'release_month' as well
	df.drop(['release_date'], axis=1)

	# build features out of statistics on numerical variable
	original_features = list(df.columns)
	original_features.remove('revenue')
	group_by_variables = ['release_season','got_img','collection_id','production_company_id','director_id','producer_id']
	if group_by_x:
		group_by_variables=[group_by_x]

	df = build_statistics_features(df,group_by_variables,original_features,
							  ['production_company_country','genres','top_3_actors']
							  +[col for col in df.columns if '_id' in col])

	return df

if __name__=='__main__':
	# train = pd.read_csv('data/train.tsv', sep="\t")
	# test = pd.read_csv('data/test.tsv', sep="\t")
	# df = preprocess_shir(train,test,True)

	train = pd.read_csv('./data/clean_train.csv')
	test = pd.read_csv('./data/clean_test.csv')
	# preparing_data_to_expirement(train,test)

	integration(train,test)



	# data = pd.read_csv('data/train.tsv', sep="\t")
	# df = preprocess_shir(data)

	# df = pd.read_csv('./data/clean_data.csv')
	# df_built = build_features(df)
	# rich_dfs = [[None,'genres'],[None,'top_3_actors']]
	# for rich_df in rich_dfs:
	# 	rich_df[0] = enrich_dataset_by_dividing_x(df,rich_df[1])
	# 	rich_df[0].to_csv('./data/rich_df_'+rich_df[1]+'.csv',index=False,header=True)
	# 	rich_df[0] = build_features(rich_df[0],group_by_x=rich_df[1])
	# 	rich_df[0] = rich_df[0].groupby('id').agg(['mean'])
	# 	relevant_columns = [idx for idx,col in enumerate(rich_df[0].columns) if rich_df[1] in col[0]]
	# 	rich_df[0] = rich_df[0].iloc[:,relevant_columns]
	# 	df_built = df_built.merge(rich_df[0], on='id', how='left')
	# df,cols_to_remove = delete_highly_target_correlated_features(df_built,'revenue')
	# print(df.shape)
