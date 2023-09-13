#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 18:26:46 2023

@author: thanuja
"""

import tensorflow as tf

import numpy as np
import pandas as pd

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'
max_tokens = None
#embedding_dim = 20
max_length = 50

train_df = pd.read_csv(base_dir + 'test_data_raw.features.csv').fillna(0)
sentences = train_df['original_text'].to_numpy()
X_train = train_df['pos_seq'].astype('string').to_numpy()

X_train = tf.convert_to_tensor(X_train)

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

model = tf.keras.models.load_model(base_dir + 'gru.model')

model.summary()

predictions = [p[0] for p in model.predict(x=X_train)]


predictions_df = pd.read_csv(base_dir + 'supervised_predictions.csv')
#predictions_df['GRU on PoS'] = predictions
#

pred_map = {}

for i in range(X_train.shape[0]):
    #print(predictions[i][0], X_train[i].numpy())
    pred_map[sentences[i].lower()] = predictions[i]

skipped = 0
total = 0

for index, row in predictions_df.iterrows():
    sentence = row['sentence'].lower()
    if total % 10000 == 0: print(total)
    total += 1
    if sentence not in pred_map:
        skipped += 1
        continue
    #print('index', index)
    #print('row', row)
    pred = pred_map[row['sentence'].lower()]
    predictions_df.loc[index, 'GRU on PoS'] = pred
    
print('skipped', skipped)
print('total', total)
print(predictions_df.head())

predictions_df.to_csv(base_dir + 'supervised_predictions.csv')