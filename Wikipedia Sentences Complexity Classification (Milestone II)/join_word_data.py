#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:10:45 2023

@author: thanuja
"""

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'

import pandas as pd
from collections import deque
from collections import defaultdict

parts_of_speech = {'Verb', 'Noun', 'Determiner', 'ProperNoun',
                   'Preposition', 'Conjunction', 'Adjective', 'Article',
                   'UnknownPoS', 'Pronoun', 'Adverb'}

########### Age of Aquisition #################
aoa_df = pd.read_csv(base_dir + 'AoA_51715_words.csv', encoding = 'ISO-8859-1')

aoa_df = aoa_df[['Word', 'Alternative.spelling', 'Freq_pm', 'Dom_PoS_SUBTLEX',
                 'Nletters', 'Nsyll', 'Lemma_highest_PoS', 'Perc_known',
                 'AoA_Kup_lem']]

word_map = {}

for index, row in aoa_df.iterrows():
    word_data = {'simple': False}
    for col in ['Word', 'Freq_pm', 'Dom_PoS_SUBTLEX', 'Nletters', 'Nsyll',
                 'Lemma_highest_PoS', 'Perc_known', 'AoA_Kup_lem']:
        word_data[col] = row[col]
    word = word_data['Word']
    if type(word) is not str:
        #print('What is: ', word, type(word), row)
        continue
    word_map[word.lower()] = word_data
    if word != row['Alternative.spelling']:
        word_map[row['Alternative.spelling'].lower()] = word_data

########### Concreteness #################
concreteness_df = pd.read_csv(base_dir + 'Concreteness_ratings_Brysbaert_et_al_BRM.txt',
                              delimiter = '\t')
print(concreteness_df.head())

for index, row in concreteness_df.iterrows():
    if row['Bigram'] == 1:
        # avoid Bigrams for simplicity in sentence tokenization
        continue
    word = row['Word']
    if type(word) is not str:
        continue
    word = word.strip().lower()
    if word in word_map:
        word_data = word_map[word]
    else:
        word_data = {'simple': False}
    
    for col in ['Conc.M', 'Conc.SD']:
        word_data[col] = row[col]
    if not 'Dom_PoS_SUBTLEX' in word_data:
        word_data['Dom_PoS_SUBTLEX'] = row['Dom_Pos']
    word_map[word] = word_data

#print(word_data)

########## Update related words ###########
for word, word_data in word_map.items():
    # Lemma_highest_PoS - is similar to base word eg. imitating: imitate (base word)
    if 'Lemma_highest_PoS' not in word_data:
        continue
    base_word = word_data['Lemma_highest_PoS']
    if type(base_word) is str and word != base_word.lower():
        base_word_data = word_map[base_word.lower()]
        word_data['Perc_known'] = base_word_data['Perc_known']

########## Dale Challa simple words ###########
with open(base_dir + 'dale_chall.txt') as simple_words:
    for simple_word in simple_words:
        simple_word = simple_word.strip().lower()
        if simple_word not in word_map:
            #print('Simple word', simple_word, 'not found, skipping')
            # mainly contractions and sound words (ha, baa, ding-dong)
            continue
        word_map[simple_word]['simple'] = True


print('sovereignty', word_map['sovereignty']) # both AoA and Concreteness
print('improvable', word_map['improvable']) # just Concreteness
print('abalones', word_map['abalones']) # just AoA
print('adventure', word_map['adventure']) # simple word

#print(aoa_df.head())

def count_n_grams(pos_arr, n):
    result = set()
    q = deque()
    for pos in pos_arr:
        q.append(pos)
        if len(q) < n:
            continue
        if len(q) > n:
            q.popleft()
        result.add(tuple(q))
    return len(result)

matched_words = defaultdict(int)
unmatched_words = defaultdict(int)

def get_pos(word):
    result = None
    if word.lower() in word_map:
        word_data = word_map[word.lower()]
        if 'Dom_PoS_SUBTLEX' in word_data:
            result = word_data['Dom_PoS_SUBTLEX']
    elif word.istitle():
        result = 'ProperNoun'
    elif word.endswith('ly'):
        result = 'Adverb'
    
    if result in parts_of_speech:
        return result
    return 'UnknownPoS'

def my_mean(arr):
    if len(arr) == 0: return 0
    return sum(arr) / len(arr)

def my_max(arr):
    result = 0
    for a in arr:
        if result < a: result = a
    return result

def get_features(sentence):
    global matched_words, unmatched_words
    result = {}
    word_count = 0
    num_count = 0
    punct_count = 0
    simple_count = 0
    freqs = []
    aoas = []
    word_lens = []
    syll_counts = []
    pct_knowns = []
    conc_means = []
    conc_stds = []
    pos_seq = []
    pos_map = defaultdict(int)

    def get_word_data(word):
        nonlocal simple_count, syll_counts, pct_knowns, aoas, conc_means, conc_stds
        word_data = word_map[word.lower()]
        if word_data['simple']: simple_count += 1
        if 'Nsyll' in word_data: syll_counts.append(word_data['Nsyll'])
        if 'Perc_known' in word_data: pct_knowns.append(word_data['Perc_known'])
        if 'AoA_Kup_lem' in word_data: aoas.append(word_data['AoA_Kup_lem'])
        if 'Conc.M' in word_data: conc_means.append(word_data['Conc.M'])
        if 'Conc.SD' in word_data: conc_stds.append(word_data['Conc.SD'])
        if 'Freq_pm' in word_data: freqs.append(word_data['Freq_pm'])
        return word_data

    #debug_arr = []
    for word in sentence.split():
        if word.isnumeric():
            pos_seq.append('Num')
            num_count += 1
            continue
        elif not word.isalpha():
            pos_seq.append('Punct')
            punct_count += 1
            continue
        
        word_lens.append(len(word))
        word_count += 1
        
        pos = get_pos(word)
        pos_map[pos] += 1
        pos_seq.append(pos)
        #debug_arr.append((word, pos))
        if word.lower() in word_map:
            matched_words[word.lower()] += 1
            word_data = get_word_data(word)
        elif word.endswith('ed'):
            if word.lower()[:-2] in word_map:
                matched_words[word.lower()] += 1
                word_data = get_word_data(word[:-2])
            elif word.lower()[:-1] in word_map:
                matched_words[word.lower()] += 1
                word_data = get_word_data(word[:-1])
            elif len(word) > 3 and word.lower()[:-3] in word_map and word[-3] == word[-4]:
                # spammed -> spam
                matched_words[word.lower()] += 1
                word_data = get_word_data(word[:-3])
            else:
                unmatched_words[word.lower()] += 1
        elif word.endswith('ing'):
            if word.lower()[:-3] in word_map:
                word_data = get_word_data(word[:-3])
                matched_words[word.lower()] += 1
            elif word.lower()[:-3] + 'e' in word_map:
                # imitating -> imitate
                word_data = get_word_data(word[:-3] + 'e')
                matched_words[word.lower()] += 1
            elif len(word) > 4 and word.lower()[:-4] in word_map and word[-4] == word[-5]:
                word_data = get_word_data(word[:-4])
                matched_words[word.lower()] += 1
            else:
                unmatched_words[word.lower()] += 1
        elif pos != 'ProperNoun':
            unmatched_words[word.lower()] += 1
    if word_count == 0:
        return result
    for pos, pos_count in pos_map.items():
        result[pos] = pos_count # / word_count
    result['pos_seq'] = ' '.join(pos_seq)
    result['word_count'] = word_count
    result['num_count'] = num_count
    result['punct_count'] = punct_count
    result['uniq_1_grams'] = count_n_grams(pos_seq, 1)
    result['uniq_2_grams'] = count_n_grams(pos_seq, 2)
    result['uniq_3_grams'] = count_n_grams(pos_seq, 3)
    result['pct_simple'] = simple_count / word_count
    result['mean_syll'] = my_mean(syll_counts)
    result['mean_pct_known'] = my_mean(pct_knowns)
    result['mean_aoa'] = my_mean(aoas)
    result['max_aoa'] = my_max(aoas)
    result['mean_conc_mean'] = my_mean(conc_means)
    result['max_conc_mean'] = my_max(conc_means)
    result['mean_conc_std'] = my_mean(conc_stds)
    result['mean_freqs'] = my_mean(freqs)
    #print(debug_arr)
    return result
    

#test_sentence = 'Plays and comic puppet theater loosely based on this legend were popular throughout Germany in the 16th century , often reducing Faust and Mephistopheles to figures of vulgar fun .'
#print(get_features(test_sentence))

def read_text(filename):
    text_df = pd.read_csv(base_dir + filename)
    print('*************', filename)
    print(text_df.head())
    for index, row in text_df.iterrows():
        if index % 10000 == 0: print(index)
        sentence = row['original_text']
        features = get_features(sentence)
        for feature, value in features.items():
            text_df.at[index, feature] = value
        text_df.at[index, 'label'] = row['label']
            
    return text_df
    
test_df = read_text('WikiLarge_Test.csv')
print(test_df.head())
test_df.to_csv(base_dir + 'WikiLarge_Test.features.csv')

test_df = read_text('test_data_raw.csv')
print(test_df.head())
test_df.to_csv(base_dir + 'test_data_raw.features.csv')

train_df = read_text('training_data_raw.csv')
print(train_df.head())
train_df.to_csv(base_dir + 'training_data_raw.features.csv')

print('matched', len(matched_words))
print('unmatched', len(unmatched_words))

sorted_unmatched = sorted(unmatched_words.items(), key=lambda row: row[1],
                          reverse=True)
with open(base_dir + 'unmatched_words.txt', 'w') as unmatched_file:
    unmatched_file.writelines(w[0] + ',' + str(w[1]) + '\n'
                              for w in sorted_unmatched)
