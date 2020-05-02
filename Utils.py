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
			try:
				data[num] = pd.to_numeric(data[num], downcast='unsigned')
			except:
				pass

	return data

def delete_highly_target_correlated_features(df,target,threshold=0.8):
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
	agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min']).reset_index()
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

def preprocess(data,train=False, save=True , cols=None):
	data['log_revenue'] = np.log10(data['revenue']+1)
	data['log_budget'] = np.log10(data['budget']+1)
	data['homepage'] = np.where(data['homepage'].isna(), 0, 1)

	data['release_date'] = pd.to_datetime(data['release_date'])
	data['release_year'] = data['release_date'].dt.year
	data['release_month'] = data['release_date'].dt.month
	data['passed_years'] = date.today().year - data['release_year']
	data['release_season'] = pd.cut(data['release_month'], bins=[0, 3, 6, 9, 12],
									labels=["Winter", "Spring", "Summer", "Autumn"]).astype('category')

	# extract collection features
	data['belongs_to_collection'] = data['belongs_to_collection'].fillna('{}')
	data['collection_id'] = data['belongs_to_collection'].apply(lambda x: parser(x)['id'] if ((x != {}) and ('id' in parser(x).keys())) else 'No Collection')
	data['has_collection'] = data['collection_id'].apply(lambda x: 1 if x != 'No Collection' else 0)
	data['collection_name'] = data['belongs_to_collection'].apply(lambda x: parser(x)['name'] if ((x != {}) and ('name' in parser(x).keys())) else 0)

	data.drop(['belongs_to_collection'],axis=1,inplace=True)


	# extract genres (There are 19 of them)
	if train:
		list_of_genres = list(data['genres'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else ['Other']).values)
		set_of_genres = set([m for m in Counter([i for j in list_of_genres for i in j])])
		data['genres'] = data['genres'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else ['Other']).values
		data['num_of_geners'] = data['genres'].apply(lambda x: len(x))
		for g in set_of_genres:
			data['genre_' + g] = data['genres'].apply(lambda x: 1 if g in x else 0)
	else:
		data['genres'] = data['genres'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else ['Other']).values
		data['num_of_geners'] = data['genres'].apply(lambda x: len(x))
		for g in [col.split('_')[-1] for col in cols if 'genre_' in col]:
			data['genre_' + g] = data['genres'].apply(lambda x: 1 if g in x else 0)
	# data.drop(['genres'], axis=1, inplace=True)

	# extract production company details
	if train:
		list_of_companies = list(
			data['production_companies'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)

		data['all_production_companies'] = data['production_companies'].apply(
			lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
		data['num_companies'] = data['all_production_companies'].apply(lambda x: len(x))
		top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]
		for g in top_companies:
			data['production_company_' + g] = data['all_production_companies'].apply(lambda x: 1 if g in x else 0)
	else:
		data['all_production_companies'] = data['production_companies'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
		data['num_companies'] = data['all_production_companies'].apply(lambda x: len(x))
		for g in [col.split('_')[-1] for col in cols if 'production_company_' in col]:
			data['production_company_' + g] = data['all_production_companies'].apply(lambda x: 1 if g in x else 0)
	# data.drop(['production_companies','all_production_companies'], axis=1, inplace=True)
	data.drop(['production_companies'], axis=1, inplace=True)

	# production countries
	if train:
		list_of_countries = list(data['production_countries'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)

		data['all_countries'] = data['production_countries'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
		data['num_countries'] = data['all_countries'].apply(lambda x: len(x))
		top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(15)]
		for g in top_countries:
			data['production_country_' + g] = data['all_countries'].apply(lambda x: 1 if g in x else 0)
	else:
		data['all_countries'] = data['production_countries'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
		data['num_countries'] = data['all_countries'].apply(lambda x: len(x))
		for g in [col.split('_')[-1] for col in cols if 'production_country_' in col]:
			data['production_country_' + g] = data['all_countries'].apply(lambda x: 1 if g in x else 0)

	# data.drop(['production_countries', 'all_countries'], axis=1)
	data.drop(['production_countries'], axis=1)


	# extract features from cast
	if train:
		list_of_cast_names = list(data['cast'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)
		list_of_cast_characters = list(data['cast'].apply(lambda x: [i['character'] for i in parser(x)] if x != {} else []).values)
		data['all_cast'] = data['cast'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
		data['num_cast'] = data['all_cast'].apply(lambda x: len(x))
		top_cast_names = [str(m[0]) for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]
		for g in top_cast_names:
			data['cast_name_' + g] = data['cast'].apply(lambda x: 1 if g in str(x) else 0)
		data['genders_0_cast'] = data['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 0]))
		data['genders_1_cast'] = data['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 1]))
		data['genders_2_cast'] = data['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 2]))
		top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(10)]
		for g in top_cast_characters:
			data['cast_character_' + g] = data['cast'].apply(lambda x: 1 if g in str(x) else 0)
	else:
		data['all_cast'] = data['cast'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
		data['num_cast'] = data['all_cast'].apply(lambda x: len(x))
		for g in [col.split('_')[-1] for col in cols if 'cast_name_' in col]:
			data['cast_name_' + g] = data['cast'].apply(lambda x: 1 if g in str(x) else 0)
		data['genders_0_cast'] = data['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 0]))
		data['genders_1_cast'] = data['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 1]))
		data['genders_2_cast'] = data['cast'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 2]))
		for g in [col.split('_')[-1] for col in cols if 'cast_character_' in col]:
			data['cast_character_' + g] = data['cast'].apply(lambda x: 1 if g in str(x) else 0)
	# data.drop(['cast','all_cast'], axis=1, inplace=True)
	data.drop(['cast'], axis=1, inplace=True)


	# extract features from crew
	if train:
		list_of_crew_names = list(data['crew'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values)
		list_of_crew_jobs = list(data['crew'].apply(lambda x: [i['job'] for i in parser(x)] if x != {} else []).values)
		list_of_crew_departments  = list(data['crew'].apply(lambda x: [i['department'] for i in parser(x)] if x != {} else []).values)

		data['all_crew'] = data['crew'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
		data['num_crew'] = data['all_crew'].apply(lambda x: len(x))
		top_crew_names = [str(m[0]) for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
		for g in top_crew_names:
			data['crew_name_' + g] = data['crew'].apply(lambda x: 1 if g in str(x) else 0)
		top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(10)]
		data['genders_0_crew'] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 0]))
		data['genders_1_crew'] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 1]))
		data['genders_2_crew'] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 2]))
		for j in top_crew_jobs:
			data['jobs_' + j] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['job'] == j]))
		top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(10)]
		for j in top_crew_departments:
			data['departments_' + j] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['department'] == j]))
	else:
		data['all_crew'] = data['crew'].apply(lambda x: [i['name'] for i in parser(x)] if x != {} else []).values
		data['num_crew'] = data['all_crew'].apply(lambda x: len(x))
		for g in [col.split('_')[-1] for col in cols if 'crew_name_' in col]:
			data['crew_name_' + g] = data['crew'].apply(lambda x: 1 if g in str(x) else 0)
		data['genders_0_crew'] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 0]))
		data['genders_1_crew'] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 1]))
		data['genders_2_crew'] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['gender'] == 2]))
		for j in [col.split('_')[-1] for col in cols if 'jobs_' in col]:
			data['jobs_' + j] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['job'] == j]))
		for j in [col.split('_')[-1] for col in cols if 'departments_' in col]:
			data['departments_' + j] = data['crew'].apply(lambda x: sum([1 for i in parser(x) if i['department'] == j]))
	# data.drop(['crew','all_crew'],axis=1,inplace=True)
	data.drop(['crew'],axis=1,inplace=True)

	data = integration(data)


	data['got_img'] = np.where(data['backdrop_path'].isna(),0,1)
	data['tagline'] = np.where(data['tagline'].isna(), 0, 1)
	data['video'] = np.where(data['video']==True,1,0)
	data['runtime'] = data['runtime'].fillna(np.median(data['runtime'].dropna()))
	data.drop(['backdrop_path','all_crew','all_cast','all_countries','all_production_companies','genres'], axis=1, inplace=True)


	if save:
		data.to_csv('./data/clean_integrated_data.csv',header=True,index=False)

	return data

def integration(df):
	#add productions information
	production_df = pd.read_csv("./data/production.csv",encoding="ISO-8859-1")

	production_df['total_revenue'] = production_df.total_revenue.str.replace(r'\D','').astype(np.int64)
	production_df['avg_revenue_for_movie'] = production_df.apply(lambda x: x.total_revenue / x.number_of_movies, axis=1)

	df['production_number_of_movies'] = 0
	df['production_avg_rev_for_movie'] = 0
	df['production_total_rev'] = 0

	#collections
	collectiond_df = pd.read_csv("./data/collections.csv",encoding="ISO-8859-1")
	collectiond_df['revnue'] = collectiond_df.revnue.str.replace(r'\D','').astype(np.int64)
	collectiond_df['avg_revenue_for_movie'] = collectiond_df.apply(lambda x: x.revnue / x.number_of_movies, axis=1)
	collectiond_df['collection'] = collectiond_df['collection'].str.lower()

	df['collection_name'] = df['collection_name'].str.lower()
	df['collection_name'] = df['collection_name'].str.replace(r' collection', '')

	df['collection_number_of_movies'] = 0
	df['collection_avg_rev_for_movie'] = 0
	df['collection_total_rev'] = 0

	#years
	years_df = pd.read_csv("./data/years.csv",encoding="ISO-8859-1")
	years_df = years_df.fillna('0')
	years_df['number_of_movies'] = years_df.number_of_movies.str.replace(r'\D','').astype(np.int64)
	years_df['total_revenue'] = years_df.total_revenue.str.replace(r'\D','').astype(np.int64)
	years_df['avg_revenue_for_movie'] = years_df.apply(lambda x: x.total_revenue / x.number_of_movies, axis=1)

	df['year_number_of_movies'] = 0
	df['year_avg_rev_for_movie'] = 0
	df['year_total_rev'] = 0

	#rating
	rating_df = pd.read_csv('./data/rating.tsv', sep="\t")

	df['rating_number_of_votes'] = 0
	df['rating_avg'] = 0


	for index, row in df.iterrows():
		#production
		production_companies = row['all_production_companies']
		rows_prod = production_df.loc[production_df['production'].isin(production_companies)]
		number_of_movies = list(rows_prod['number_of_movies'])
		avg_rev_for_movie = list(rows_prod['avg_revenue_for_movie'])
		production_total_rev = list(rows_prod['total_revenue'])
		if number_of_movies:
			df['production_number_of_movies'][index] = max(number_of_movies)
			df['production_avg_rev_for_movie'][index] = max(avg_rev_for_movie)
			df['production_total_rev'][index] = max(production_total_rev)

		#collections
		collection = row['collection_name']
		rows_collection = collectiond_df.loc[collectiond_df['collection'] == collection]
		if not rows_collection.empty:
			df['collection_number_of_movies'][index] = list(rows_collection['number_of_movies'])[0]
			df['collection_avg_rev_for_movie'][index] = list(rows_collection['avg_revenue_for_movie'])[0]
			df['collection_total_rev'][index] = list(rows_collection['revnue'])[0]

		#years
		year = row['release_year']
		rows_year = years_df.loc[years_df['year'] == year]
		if not rows_year.empty:
			rows_year = rows_year.fillna(0)
			df['year_number_of_movies'][index] = list(rows_year['number_of_movies'])[0]
			df['year_avg_rev_for_movie'][index] = list(rows_year['avg_revenue_for_movie'])[0]
			df['year_total_rev'][index] = list(rows_year['total_revenue'])[0]

		#rating
		imdb = row['imdb_id']
		rows_rating = rating_df.loc[rating_df['tconst'] == imdb]
		if not rows_rating.empty:
			rows_rating = rows_rating.fillna(0)
			df['rating_avg'][index] = list(rows_rating['averageRating'])[0]
			df['rating_number_of_votes'][index] = list(rows_rating['numVotes'])[0]


	df.to_csv('./data/itegrated_data.csv', header=True, index=False)
	return df
	print()


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
	# return build_statistics_features_on_categorical_vars(df, group_by_variables, original_features ,exclude)
	return df

def build_features(df,cat_vars,numerical_vars,group_by_x=None):
	df = shrink_memory_consumption(df,cat_vars,numerical_vars)

	# build features out of statistics on numerical variable
	original_features = list(df.columns)
	group_by_variables = ['release_season','collection_id','original_language','jobs_Executive Producer','jobs_Art Direction']
	if group_by_x:
		group_by_variables=[group_by_x]

	df = build_statistics_features(df,group_by_variables,original_features,
							  ['production_company_country','genres','top_3_actors']
							  +[col for col in df.columns if '_id' in col])

	return df

if __name__=='__main__':
	raw_data = pd.read_csv('data/train.tsv', sep="\t")
	df = preprocess(raw_data,train=True)
	# data = pd.read_csv('./data/clean_data.csv')

	# train = pd.read_csv('./data/clean_train.csv')
	# test = pd.read_csv('./data/clean_test.csv')
	# preparing_data_to_expirement(train,test)

	# integration(train)


	# df_built = build_features(data)
	# rich_dfs = [[None,'genres'],[None,'top_3_actors']]
	# for rich_df in rich_dfs:
	# 	rich_df[0] = enrich_dataset_by_dividing_x(data, rich_df[1])
	# 	rich_df[0].to_csv('./data/rich_df_'+rich_df[1]+'.csv',index=False,header=True)
	# 	rich_df[0] = build_features(rich_df[0],group_by_x=rich_df[1])
	# 	rich_df[0] = rich_df[0].groupby('id').agg(['mean'])
	# 	relevant_columns = [idx for idx,col in enumerate(rich_df[0].columns) if rich_df[1] in col[0]]
	# 	rich_df[0] = rich_df[0].iloc[:,relevant_columns]
	# 	df_built = df_built.merge(rich_df[0], on='id', how='left')
	# data, cols_to_remove = delete_highly_target_correlated_features(df_built, 'revenue')
	# print(data.shape)
	# _=0