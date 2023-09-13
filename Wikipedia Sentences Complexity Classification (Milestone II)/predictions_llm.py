#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 18:26:46 2023

@author: thanuja
"""

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'

from transformers import AutoTokenizer, DataCollatorWithPadding, create_optimizer, TFAutoModelForSequenceClassification
from datasets import load_dataset, Dataset
import evaluate

import numpy as np
import pandas as pd
import tensorflow as tf

dataset = 'test_data_raw.csv'
model_dir = base_dir + 'llm.model'
config = 'distilbert-base-uncased-finetuned-sst-2-english'
batch_size = 32

tokenizer = AutoTokenizer.from_pretrained(config)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

wiki_df = pd.read_csv(base_dir + dataset) #.sample(1000)
sentences = wiki_df['original_text'].to_numpy()
print('#sentences', sentences.shape[0])

train_sentences = Dataset.from_pandas(wiki_df)
print('raw sentences', train_sentences)
tokenized_train_sentences = train_sentences.map(lambda s: tokenizer(s['original_text'],
                                                                    truncation=True))

print('processed sentences', tokenized_train_sentences)

model = TFAutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.summary()
#model = tf.keras.models.load_model(model_dir)

tf_train_set = model.prepare_tf_dataset(
    tokenized_train_sentences,
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)
predictions = tf.nn.softmax(model.predict(x=tf_train_set).logits).numpy()

print('predictions', predictions)

predictions = [p[1] for p in predictions]

print('#predictions', len(predictions))

pred_map = {}

for i in range(sentences.shape[0]):
    #print(predictions[i][0], sentences[i].numpy())
    pred_map[sentences[i].lower()] = predictions[i]
print('type', type(predictions[0]))

predictions_df = pd.read_csv(base_dir + 'supervised_predictions.csv')

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
    predictions_df.loc[index, 'Distilbert'] = pred
    
print('skipped', skipped)
print('total', total)
print(predictions_df.head())

predictions_df.to_csv(base_dir + 'supervised_predictions.csv')
