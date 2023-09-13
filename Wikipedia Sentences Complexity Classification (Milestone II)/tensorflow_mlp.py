#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 21:56:00 2023

@author: thanuja
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'
min_df=100

def load_bag_of_words():
    global base_dir, min_df
    train_df = pd.read_csv(base_dir + 'WikiLarge_Train.csv')
    X_train = train_df.to_numpy()
    X_train_sentences = np.squeeze(X_train[:,:1])
    y_train = np.squeeze(X_train[:,-1:])
    
    test_df = pd.read_csv(base_dir + 'WikiLarge_Test.csv')
    X_test = test_df.to_numpy()
    X_test_sentences = np.squeeze(X_test[:,1:2])
    
    #all_sentences = np.concatenate((X_train_sentences, X_test_sentences), axis=0)
    #word_bagger = CountVectorizer(min_df=min_df)
    word_bagger = TfidfVectorizer(min_df=min_df)
    word_bagger.fit(X_train_sentences)
    print('number of unique words', len(word_bagger.vocabulary_))
    # print(word_bagger.vocabulary_)
    #print('stop words', word_bagger.stop_words_)
    # Default: number of unique words 157615
    # min_df=2, number of unique words 122930
    # min_df=3, number of unique words 86516
    # min_df=4, number of unique words 73995
    # min_df=100, number of unique words 6451
    
    X_train = word_bagger.transform(X_train_sentences)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=0.1, random_state=0)
    y_train = y_train.astype('int')
    y_validation = y_validation.astype('int')
    #print(X_train_sentences[:10])
    
    X_test = word_bagger.transform(X_test_sentences)
    
    return X_train, X_validation, y_train, y_validation, X_test

def load_features():
    global base_dir
    train_df = pd.read_csv(base_dir + 'WikiLarge_Train.features.csv').fillna(0)
    
    X_train = train_df.drop(columns=['label']).to_numpy()[:,2:]
    print(X_train[:2])
    enc = OneHotEncoder(handle_unknown='ignore')
    X_train = enc.fit_transform(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = np.squeeze(train_df['label'].to_numpy()).astype('int')
    
    test_df = pd.read_csv(base_dir + 'WikiLarge_Test.features.csv').fillna(0)
    X_test = test_df.drop(columns=['label']).to_numpy()[:,3:]
    X_test = enc.transform(X_test)
    print(X_test[:2])
    X_test = scaler.transform(X_test)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=0.1, random_state=0)
    return X_train, X_validation, y_train, y_validation, X_test

def load_embeddings():
    global base_dir
    train_df = pd.read_csv(base_dir + 'WikiLarge_Train.embeddings.csv').fillna(0)
    
    X_train = train_df.drop(columns=['label']).to_numpy()[:,3:]
    print(X_train[:2])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = np.squeeze(train_df['label'].to_numpy()).astype('int')
    X_train, y_train = shuffle(X_train, y_train)

    '''    
    test_df = pd.read_csv(base_dir + 'WikiLarge_Test.features.csv').fillna(0)
    X_test = test_df.drop(columns=['label']).to_numpy()[:,3:]
    X_test = scaler.transform(X_test)
    '''
    # FIXME
    X_test = None

    return X_train, y_train, X_test

def load_features_and_embeddings():
    global base_dir
    train_df = pd.read_csv(base_dir + 'WikiLarge_Train.embeddings.60.csv').fillna(0)
    y_train = train_df['label'].to_numpy().astype('int')
    train_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'label', 'pos_seq', 'original_text'], inplace=True)
    print('feature and embedding columns', train_df.columns)

    X_train = train_df.to_numpy()
    print('X_train', X_train[:1])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    '''    
    test_df = pd.read_csv(base_dir + 'WikiLarge_Test.features.csv').fillna(0)
    X_test = test_df.drop(columns=['label']).to_numpy()[:,3:]
    X_test = scaler.transform(X_test)
    '''
    # FIXME
    X_test = None

    return X_train, y_train, X_test, train_df.columns

import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers


def tutorial():
    # https://www.kaggle.com/code/hassanamin/tensorflow-mnist-gpu-tutorial
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])
    
    predictions = model(x_train[:1]).numpy()
    predictions
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5)
    
    model.evaluate(x_test,  y_test, verbose=2)

#tutorial()


X_train, y_train, X_test, columns = load_features_and_embeddings()
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                test_size=0.1, random_state=0)

# https://github.com/google/jax/issues/13504#issuecomment-1346437937
print("executing TF bug workaround")
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8) )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# MLP example from https://www.turing.com/kb/multilayer-perceptron-in-tensorflow
# number of unique words 86516
# input_size=86516
model = models.Sequential([
    #layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='relu')
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
      loss='binary_crossentropy',
      metrics=['accuracy'])

# https://stackoverflow.com/a/43130239
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))
    #return tf.SparseTensor(indices, coo.data, coo.shape)

#X_train_tf = convert_sparse_matrix_to_sparse_tensor(X_train)
X_train_tf = X_train.astype('float32')
model.fit(X_train_tf, y_train, epochs=50, batch_size=1000) #, validation_split=0.2)
results = model.evaluate(X_validation.astype('float32'), y_validation, verbose = 1)
print('test loss, test acc:', results)

























'''
# number of unique words 86516
input_size=86516
model = models.Sequential([
    layers.Dense(1000, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(1, activation='relu')
    ])

model.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
'''