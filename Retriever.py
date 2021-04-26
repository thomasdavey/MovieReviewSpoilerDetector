import pandas as pd

id = input('Enter Movie ID: ')
name = input('Enter Movie Name: ')

data = pd.read_json('Dataset/IMDB_reviews.json', lines=True)
data = data[data.movie_id == id]
data = data.sample(frac=1)

divider = int(len(data)*0.9)
training = data[:divider]
test = data[divider:]

training.to_csv('Dataset/' + name + '_training.csv', index=False)
test.to_csv('Dataset/' + name + '_testing.csv', index=False)

data = pd.read_json('Dataset/IMDB_movie_details.json', lines=True)
data = data[data.movie_id == id]

data.to_csv('Dataset/' + name + '_synopsis.csv')