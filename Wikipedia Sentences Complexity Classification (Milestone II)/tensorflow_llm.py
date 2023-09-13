#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:41:34 2023

@author: thanuja

Based heavily on https://huggingface.co/docs/transformers/tasks/sequence_classification

"""

# 0.7690 in mixed case after 1 epoch

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'

from transformers import AutoTokenizer, DataCollatorWithPadding, create_optimizer, TFAutoModelForSequenceClassification
from datasets import load_dataset, Dataset
import evaluate

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

import csv

from transformers.keras_callbacks import KerasMetricCallback

config = 'distilbert-base-uncased-finetuned-sst-2-english'
dataset = 'training_data_raw.csv'
#dataset = 'WikiSample_Train.csv'
batch_size = 32
num_epochs = 1 # any more than 2 and it overfits
#test_size = 0.1
num_splits = 5

tokenizer = AutoTokenizer.from_pretrained(config)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

cv_results = np.empty(0)

accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    global cv_results
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result = accuracy.compute(predictions=predictions, references=labels)
    #print('type of result is', type(result['accuracy']), result['accuracy'])
    cv_results = np.append(cv_results, [result['accuracy']], axis=0)
    return result

def train_model(train_df, validation_df):
    train_sentences = Dataset.from_pandas(train_df)
    validation_sentences = Dataset.from_pandas(validation_df)
    print('raw sentences', train_sentences)
    tokenized_train_sentences = train_sentences.map(lambda s: tokenizer(s['original_text'],
                                                                        truncation=True))
    tokenized_validation_sentences = validation_sentences.map(lambda s: tokenizer(s['original_text'],
                                                                                  truncation=True))
    
    model = TFAutoModelForSequenceClassification.from_pretrained(
        config, num_labels=2
    )
   
    tf_train_set = model.prepare_tf_dataset(
        tokenized_train_sentences,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    tf_validation_set = model.prepare_tf_dataset(
        tokenized_validation_sentences,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
    
    # original lr = 2e-5
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5))
    
    model.fit(x=tf_train_set, validation_data=tf_validation_set,
              epochs=num_epochs, callbacks=[metric_callback])
    return model

def cross_validated_score():
    global cv_results
    wiki_df = pd.read_csv(base_dir + dataset) #.sample(20000)
    wiki_df['original_text'] = wiki_df['original_text'].str.lower()
    shuffle(wiki_df)
    wiki_df.reset_index(inplace=True, drop=True)
    
    ## Cross fold validation score
    cv_results = np.empty(0)
    for train_indices, validation_indices in KFold(num_splits).split(wiki_df):
        train_df = wiki_df.iloc[train_indices]
        validation_df = wiki_df.iloc[validation_indices]
        model = train_model(train_df, validation_df)
    
    print(cv_results)
    
    with open(base_dir + 'supervised_results.csv', 'a', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['Raw text', 'LLM', cv_results.mean(), cv_results.std()])
    
    ## Train on all data and save.
    model = train_model(wiki_df, wiki_df.sample(1))
    model.save_pretrained(base_dir + 'llm.model')

def learning_curve():
    global cv_results
    ## Create learning curve, use original WikiLarge_Train.csv so that it's large enough
    with open(base_dir + 'llm_learning_curve.csv', 'w', newline='') as result_file:
        wiki_df = pd.read_csv(base_dir + 'WikiLarge_Train.csv') #.sample(20000)
        wiki_df['original_text'] = wiki_df['original_text'].str.lower()
        #shuffle(wiki_df)
        #wiki_df.reset_index(inplace=True, drop=True)
        writer = csv.writer(result_file)
        data_sizes = [100 * 2**i for i in range(13)]
        writer.writerow(['Num Sentences', 'Accuracy'])
        for data_size in data_sizes:
            cv_results = np.empty(0)
            train_df, validation_df = train_test_split(wiki_df, train_size=data_size, test_size=5000)
            model = train_model(train_df, validation_df)
            print('******', data_size, cv_results.mean())
            writer.writerow([data_size, cv_results.mean()])

learning_curve()
cross_validated_score()
