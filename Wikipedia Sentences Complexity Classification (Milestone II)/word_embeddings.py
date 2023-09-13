#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 18:38:46 2023

@author: thanuja
"""

import pandas as pd
import numpy as np

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
#from sklearn.utils import shuffle

from sklearn.feature_extraction.text import TfidfVectorizer

import os.path

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'
#vector_size = 60
min_df = 1


def create_word_embeddings(train_df, vector_size):
    train_df['original_text'] = train_df['original_text'].str.lower()
    X_train = train_df.to_numpy()
    print('X_train', X_train[:2])
    print('original columns', type(list(train_df.columns)), len(train_df.columns), list(train_df.columns))
    X_train_sentences = train_df['original_text'].to_numpy()
    #print(X_train_sentences)

    print('Creating word embeddings')
    corpus = []
    for sentence in train_df['original_text']:
        corpus.append(sentence.split())
        
    model = gensim.models.Word2Vec(corpus, min_count = 1, vector_size = vector_size,
                                   window = 5, sg = 1, workers = 32)
    wv = model.wv
    print('Word embeddings created')
    #print(wv['computer'])
    #print('similarity', wv.most_similar('computer', topn=10))

    tfidf = TfidfVectorizer(min_df=min_df)
    tfidf.fit(X_train_sentences)

    return wv, tfidf

def apply_word_embeddings(X_df, tfidf, wv, vector_size):
    X_train = X_df.to_numpy()
    X_df['original_text'] = X_df['original_text'].str.lower()
    X_train_sentences = X_df['original_text'].to_numpy()
    tfidf_X = tfidf.transform(X_train_sentences)
    
    #print(tfidf_X[:2])
    new_X = np.zeros((len(X_train), vector_size))
    #print('new_X.shape', new_X.shape)
    #print('new_X[0].shape', new_X[0].shape)
    
    for i in range(0, len(X_train)):
        #print(i, X_train[i])
        #print(tfidf_X[i])
        if i % 10000 == 0: print(i)
        for w in X_train_sentences[i].split():
            if not w in tfidf.vocabulary_:
                #print('could not find', w)
                continue
            term_index = tfidf.vocabulary_[w]
            weight = tfidf_X[i,term_index]
            embedding = wv[w]
            new_X[i] += weight * embedding
            #print(i, w, term_index, weight, embedding, sentence_vector)
    
    new_X = np.concatenate((X_train, new_X), axis=1)
    #print(new_X.shape)
    
    final_df = pd.DataFrame(new_X, columns = list(X_df.columns)
                            + ['we_' + str(i) for i in range(0, vector_size)])
    #print(final_df.head())
    return final_df

'''
for vector_size in range(10, 151, 10):
    print(vector_size)
    create_word_embeddings(vector_size)
'''

vector_size = 20

# Split into create/apply word embeddings
train_df = pd.read_csv(base_dir + 'training_data_raw.features.csv')
test_df = pd.read_csv(base_dir + 'test_data_raw.features.csv')
test_df = test_df[train_df.columns]

wv, tfidf = create_word_embeddings(train_df, vector_size)
df = apply_word_embeddings(train_df, tfidf, wv, vector_size)
df.to_csv(base_dir + 'training_data_raw.embeddings.20.csv')

# Reuse same word embeddings for test, or test predictions will fail.
test_df['original_text'] = train_df['original_text'].str.lower()

df = apply_word_embeddings(test_df, tfidf, wv, vector_size)
df.to_csv(base_dir + 'test_data_raw.embeddings.20.csv')

#df = create_word_embeddings(base_dir + 'WikiLarge_Test.features.csv', 20)
#df.to_csv(base_dir + 'WikiLarge_Test.embeddings.20.csv')


'''
    word_vec_file = base_dir + 'X_train.' + str(vector_size) + '.wordvectors'
    wv = None
    
    if not os.path.isfile(word_vec_file): 
        corpus = []
        for sentence in train_df['original_text']:
            corpus.append(sentence.split())
            
        model = gensim.models.Word2Vec(corpus, min_count = 1, vector_size = vector_size,
                                       window = 5, sg = 1, workers = 32)
    
        model.wv.save(word_vec_file)
        wv = model.wv
    else:
        wv = KeyedVectors.load(word_vec_file, mmap='r')

'''