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
    
    print('Initialising...')
    
    documents = pd.read_csv('Dataset/' + title + '_training.csv', error_bad_lines=False)
    documents = documents[['review_text', 'is_spoiler']]
    documents = documents.assign(index=documents.index)

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
 
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    print('Initialised')
    print('Loading LDA Model...')
    
    lda_model = gensim.models.LdaMulticore.load('Exports/' + title + '_bow.model')
    
    synopsis = pd.read_csv('Dataset/' + title + '_synopsis.csv', error_bad_lines=False)
    synopsis = synopsis[['plot_synopsis']].to_string()
    
    bow_vector = dictionary.doc2bow(preprocess(synopsis))
    
    print('\nSynopsis Similarity\n')
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\nTopic {}: {}\n".format(score, index, lda_model.print_topic(index, 10)))           
    
    is_spoiler = 0
    is_spoiler_count = 0
    not_spoiler = 0
    not_spoiler_count = 0
    
    scores = []    
    
    documents = documents.to_dict()
    
    for index, spoiler in documents['is_spoiler'].items():
        score = gensim.matutils.cossim(lda_model[bow_vector], lda_model[bow_corpus[index]])
        scores.append((spoiler, score, index))
        if spoiler == True:
            is_spoiler += score
            is_spoiler_count += 1
        else: 
            not_spoiler += score
            not_spoiler_count += 1    

    print('Spoiler Count:\t{}'.format(is_spoiler_count))
    print('Spoiler Score:\t{}'.format(is_spoiler))
    print('Average:\t{}\n'.format(is_spoiler/is_spoiler_count))
    print('Non-Spoiler Count:\t{}'.format(not_spoiler_count))
    print('Non-Spoiler Score:\t{}'.format(not_spoiler))
    print('Average:\t\t{}\n'.format(not_spoiler/not_spoiler_count))    
    print('\nScore\t{}\n'.format(round((((is_spoiler/is_spoiler_count)-(not_spoiler/not_spoiler_count))*100),2)))    
    
    print('\nDocument Scores\n')
    count = 0
    for spoiler, score, index in sorted(scores, key=lambda tup: -1*tup[1]):
        if count >= 50:
            break
        count += 1
        print('{}\t{}\t{}\t{}'.format(count, spoiler, score, index))  







