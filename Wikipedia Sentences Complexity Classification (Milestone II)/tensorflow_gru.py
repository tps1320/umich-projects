#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:07:56 2023

@author: thanuja

Based on https://regenerativetoday.com/implementation-of-simplernn-gru-and-lstm-models-in-keras-and-tensorflow-for-an-nlp-project/
and https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization

"""

import tensorflow as tf

import numpy as np
import pandas as pd

import evaluate
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

import csv

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'
max_tokens = None
max_length = 50
num_epochs=7

dataset = 'training_data_raw.features.csv'
num_splits = 5

cv_results = np.empty(0)

accuracy = evaluate.load('accuracy')

# from https://www.tensorflow.org/guide/keras/custom_callback
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and accuracy is {:7.2f}.".format(
                epoch, logs["loss"], logs["accuracy"]
            )
        )

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))
        global cv_results
        #print('type of result is', type(result['accuracy']), result['accuracy'])
        cv_results = np.append(cv_results, [logs['val_accuracy']], axis=0)

wiki_df = pd.read_csv(base_dir + dataset) #.sample(2000)
wiki_df['original_text'] = wiki_df['original_text'].str.lower()
shuffle(wiki_df)
wiki_df.reset_index(inplace=True, drop=True)

def train_model(train_df, validation_df):
    X_train = train_df['pos_seq'].fillna('').astype('string').to_numpy()
    y_train = train_df['label'].to_numpy()
    X_validation = validation_df['pos_seq'].fillna('').astype('string').to_numpy()
    y_validation = validation_df['label'].to_numpy()
    
    print('X_train', X_train[:2])
    print('y_train', y_train[:2])
    
    X_train = tf.convert_to_tensor(X_train)
    X_validation = tf.convert_to_tensor(X_validation)
    y_train = tf.convert_to_tensor(y_train)
    y_validation = tf.convert_to_tensor(y_validation)
    
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=None,
        split='whitespace',
        ngrams=None,
        output_mode='int',
        output_sequence_length=max_length,
        pad_to_max_tokens=False,
        vocabulary=None,
        idf_weights=None,
        sparse=False,
        ragged=False,
        encoding='utf-8'
    )
    
    tokenizer.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        tokenizer,
        tf.keras.layers.Embedding(max_length, 30,
                                  embeddings_initializer='identity', trainable=False),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50, unroll=False)),
        #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, unroll=False)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
        
    history=model.fit(X_train, y_train, epochs=num_epochs,
                      validation_data = (X_validation, y_validation),
                      callbacks=[CustomCallback()])
    return model

def learning_curve():
    global cv_results
    ## Create learning curve, use original WikiLarge_Train.csv so that it's large enough
    with open(base_dir + 'gru_learning_curve.csv', 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        data_sizes = [100 * 2**i for i in range(13)]
        writer.writerow(['Num Sentences', 'Accuracy'])
        for data_size in data_sizes:
            cv_results = np.empty(0)
            train_size = min(data_size, 370000)
            train_df, validation_df = train_test_split(wiki_df, train_size=train_size, test_size=5000)
            model = train_model(train_df, validation_df)
            print('******', data_size, cv_results.mean())
            writer.writerow([data_size, cv_results.mean()])

learning_curve()

for train_indices, validation_indices in KFold(num_splits).split(wiki_df):
    train_df = wiki_df.iloc[train_indices]
    validation_df = wiki_df.iloc[validation_indices]
    model = train_model(train_df, validation_df)

print(cv_results)

with open(base_dir + 'supervised_results.csv', 'a', newline='') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(['PoS Sequences', 'GRU', cv_results.mean(), cv_results.std()])

model = train_model(wiki_df, wiki_df.sample(1))
model.save(base_dir + 'gru.model')
