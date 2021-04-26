## Movie Review Spoiler Detection Application

This is an application that adds a 'potential spoiler' warning to online movie and TV show reviews. It uses natural language processing and machine learning techniques to train a model that detects potential
spoilers in a collection of IMDB movie reviews from Kaggle, available [here](https://www.kaggle.com/rmisra/imdb-spoiler-dataset/). It was developed in Python and uses latent dirichlet allocation. It was created as part of CSC 620 at SFSU.

To run this application:

1. Find a movie ID in the IMDB_movie_details.json file.
2. Run Retriever.py and input the movie ID and nickname.
3. Run Indexer.py and input the nickname.
4. Run Evaluator.py to see how the system performs.
