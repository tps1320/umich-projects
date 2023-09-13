#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 12:43:37 2023

@author: thanuja
"""

import pandas as pd

from sklearn.model_selection import train_test_split

dataset = 'WikiLarge_Train.csv'
test_size = 0.1

wiki_df = pd.read_csv(base_dir + dataset)

train_df, test_df = train_test_split(wiki_df, test_size=test_size)

train_df.to_csv(base_dir + 'training_data_raw.csv')
test_df.to_csv(base_dir + 'test_data_raw.csv')
