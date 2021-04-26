if __name__ == '__main__':
    
    '''
    ----------------------------------------------------------------------------------------------------
    Created with the help of an article by Susan Li (Towards Data Science, 2018):
    https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
    ----------------------------------------------------------------------------------------------------
    '''
    
    import pandas as pd
    import gensim
    import nltk
    
    title = input('Enter Movie Title: ')
    
    print('Reading File...')
    
    documents = pd.read_csv('Dataset/' + title + '_training.csv', error_bad_lines=False)
    documents = documents[['review_text', 'is_spoiler']]
    documents = documents.assign(index=documents.index)
    
    print('File Successfully Read')
    print('Processing Words...')

    def lemmatize_stemming(text):
        stemmer = nltk.stem.SnowballStemmer('english')
        return stemmer.stem(nltk.stem.WordNetLemmatizer().lemmatize(text, pos='v'))
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS:
                result.append(lemmatize_stemming(token))
        return result

    processed_docs = documents['review_text'].map(preprocess)
    
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.7, keep_n=None)
    
    print('Words Proccessed')
    print('Creating BOW Corpus...')
 
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    print('BOW Corpus Created')
    print('Creating LDA Model...')
    
    num_topics = int(input('Input Number of Topics: '))
    num_passes = int(input('Input Number of Passes: '))
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=num_passes, workers=4)
    
    print('LDA Model Created')
    print('Saving Model...')
    
    lda_model.save('Exports/' + title + '_bow.model')
    
    print('Model Saved')
    print('\nTopics Created:\n')
    
    for id, topic in lda_model.print_topics(-1):
        print('Topic: {} Words: {}\n'.format(id, topic))
        