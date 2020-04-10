import pandas as pd
import numpy as np
import ast


def preprocess(data):
	data.drop(['backdrop_path','imdb_id','poster_path','original_title','tagline','id','production_countries',
			   'title','status','spoken_languages','Keywords'],axis=1,inplace=True)
	parser = ast.literal_eval

	data['homepage'] = np.where(data['homepage'].isna(),0,1)
	data['video'] = np.where(data['video']=='True',1,0)

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

if __name__=='__main__':
	raw_data = pd.read_csv('data/train.tsv', sep="\t")
	df = preprocess(raw_data)
	_=0