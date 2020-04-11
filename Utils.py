from ast import literal_eval as parser
import scipy.stats as ss
from collections import Counter
import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def preprocess(data):
	data.drop(['backdrop_path','imdb_id','poster_path','original_title','tagline','id','production_countries',
			   'title','status','spoken_languages','Keywords'],axis=1,inplace=True)

	data['homepage'] = np.where(data['homepage'].isna(),0,1)
	data['video'] = np.where(data['video']==True,1,0)
	data['budget'] = np.where(data['budget'] == 0, 1, data['budget'])
	data['vote_count'] = np.where(data['vote_count'] == 0, 1, data['vote_count'])
	data['runtime'] = data['runtime'].fillna(np.median(data['runtime'].dropna()))

	black_list = []
	for idx,num in enumerate(data['popularity']):
		if not isinstance(num,float):
			black_list.append(idx)
	data.drop(black_list,axis=0,inplace=True)


	# extract collection id
	data['belongs_to_collection'] = data['belongs_to_collection'].fillna('{}')
	uniqueAttr = set()
	data['collection_id'] = 'nan'
	for idx,cell in enumerate(data['belongs_to_collection']):
		cell = parser(cell)
		uniqueAttr = uniqueAttr.union(set(cell.keys()))
		for key,val in cell.items():
			if key == 'id':
				data['collection_id'].loc[idx] = str(val)
	data.drop(['belongs_to_collection'],axis=1,inplace=True)


	# extract genres (There are 19 of them)
	one_hot = False
	data['genres'] = data['genres'].fillna('[{}]')
	for idx,cell in enumerate(data['genres']):
		cell = parser(cell)
		combined_genre =''
		for genre in cell:
			combined_genre+=genre['name']+' '

			if one_hot:
				# one hot encoding
				if 'genre_name_'+genre['name'] not in data.columns:
					data['genre_name_' + genre['name']] = 0
				data['genre_name_'+genre['name']].loc[idx] = 1

		data['genres'].loc[idx] = combined_genre[:-1]
	# data.drop(['genres'],axis=1,inplace=True)

	# extract production company details
	data['production_company_id'] = 'nan'
	data['production_company_country'] = 'nan'
	data['production_companies'] = data['production_companies'].fillna('[{}]')
	for idx,cell in enumerate(data['production_companies']):
		cell = parser(cell)
		for element in cell:
			for key,val in element.items():
				if key=='id':
					data['production_company_id'].loc[idx] = str(val)
				elif key=='origin_country':
					data['production_company_country'].loc[idx] = val if val!='' else 'nan'
	data.drop(['production_companies'],axis=1,inplace=True)

	# extract top 3 actors
	data['top_3_actors'] = 'nan'
	data['cast'] = data['cast'].fillna('[{}]')
	for idx,cell in enumerate(data['cast']):
		cell = parser(cell)
		combined_actors = ''
		counter = 0
		for element in cell:
			if counter==3:
				break
			combined_actors += str(element['id'])+' '
			counter+=1
		data['top_3_actors'].loc[idx] = combined_actors[:-1]
	data.drop(['cast'],axis=1,inplace=True)

	# extract director and producer
	data['director_id'] = 'nan'
	data['producer_id'] = 'nan'
	data['crew'] = data['crew'].fillna('[{}]')
	for idx,cell in enumerate(data['crew']):
		cell = parser(cell)
		producer=0
		director=0
		for element in cell:
			if producer and director:
				break

			if director<1 and 'director' in element['job'].lower():
				data['director_id'].loc[idx] = element['id']
				director+=1

			if producer<1 and 'producer' in element['job'].lower():
				data['producer_id'].loc[idx] = element['id']
				producer+=1
	data.drop(['crew'],axis=1,inplace=True)

	return data

def enrich_dataset_by_dividing_x(data,x):
	data[x] = data[x].fillna('[{}]')
	for idx,cell in enumerate(data[x]):
		for element in cell.split(' '):
			new_row = data.loc[idx].to_dict()
			new_row[x]= element
			data = data.append([new_row],ignore_index = True)

	data = data[~(data[x].str.contains(' '))]
	data.reset_index(inplace=True)

	return data

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

if __name__=='__main__':
	# raw_data = pd.read_csv('data/train.tsv', sep="\t")
	# df = preprocess(raw_data)
	df = pd.read_csv('./data/rich_df_genres.csv')
	df_rich = enrich_dataset_by_dividing_x(df,'top_3_actors')
	df_rich.to_csv('./data/rich_df_actors_genres.csv',header=True,index=False)
	_=0