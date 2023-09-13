#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:52:20 2023

@author: thanuja
"""

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SequentialFeatureSelector
import scipy

import csv

min_df = 100
num_folds = 5

def load_df(file_name):
    df = pd.read_csv(base_dir + file_name)
    to_drop = []
    for col in df.columns:
        if col.startswith('Unnamed:'): to_drop.append(col)
    df.drop(columns=to_drop, inplace=True)
    return df

def load_bag_of_words():
    global base_dir, min_df
    train_df = load_df('training_data_raw.csv')
    X_train_sentences = train_df['original_text'].str.lower().to_numpy()
    y_train = train_df['label'].to_numpy()
    print('y_train', y_train[:2])
    
    test_df = load_df('test_data_raw.csv')
    X_test = test_df['original_text'].str.lower().to_numpy()
    
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
    # min_df=1000, number of unique words 809
    
    X_train = word_bagger.transform(X_train_sentences)
    y_train = y_train.astype('int')
    
    X_test_sentences = test_df['original_text'].str.lower().to_numpy()
    X_test = word_bagger.transform(X_test)
    y_test = test_df['label'].to_numpy()
    print('X_test', X_test[:2])
    
    return X_train, y_train, X_test, y_test, X_test_sentences, train_df.columns

def load_features():
    global base_dir
    train_df = load_df('training_data_raw.features.csv').fillna(0)
    y_train = train_df['label'].to_numpy().astype('int')
    train_df.drop(columns=['label', 'pos_seq', 'original_text'], inplace=True)
    
    X_train = train_df.to_numpy()
    print('X_train feature columns', train_df.columns)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    test_df = load_df('test_data_raw.features.csv').fillna(0)
    X_test_sentences = test_df['original_text'].str.lower().to_numpy()
    y_test = test_df['label'].to_numpy().astype('int')
    test_df = test_df[train_df.columns]
    #test_df.drop(columns=['label', 'pos_seq', 'original_text'], inplace=True)
    X_test = scaler.transform(test_df.to_numpy())
    print('X_test feature columns', test_df.columns)

    return X_train, y_train, X_test, y_test, X_test_sentences, train_df.columns

def load_features_and_embeddings():
    global base_dir
    train_df = load_df('training_data_raw.embeddings.20.csv').fillna(0)
    y_train = train_df['label'].to_numpy().astype('int')
    train_df.drop(columns=['label', 'pos_seq', 'original_text'], inplace=True)
    print('X_train feature and embedding columns', train_df.columns)

    X_train = train_df.to_numpy()
    print('X_train', X_train[:1])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    test_df = load_df('test_data_raw.embeddings.20.csv').fillna(0)
    X_test_sentences = test_df['original_text'].str.lower().to_numpy()
    y_test = test_df['label'].to_numpy().astype('int')
    test_df = test_df[train_df.columns]
    X_test = scaler.transform(test_df.to_numpy())
    print('X_test feature and embedding columns', test_df.columns)

    return X_train, y_train, X_test, y_test, X_test_sentences, train_df.columns

def load_embeddings():
    global base_dir
    train_df = load_df('training_data_raw.embeddings.20.csv').fillna(0)
    y_train = train_df['label'].to_numpy().astype('int')
    X_train_sentences = train_df['original_text'].str.lower().to_numpy()
    train_df = train_df[['we_' + str(i) for i in range(0, 20)]]
    print('X_train embedding columns', train_df.columns)
    X_train = train_df.to_numpy()
    print(X_train[:2])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    test_df = load_df('test_data_raw.embeddings.20.csv').fillna(0)
    X_test_sentences = test_df['original_text'].str.lower().to_numpy()
    y_test = test_df['label'].to_numpy().astype('int')
    test_df = test_df[train_df.columns]
    print('X_test embedding columns', test_df.columns)
    X_test = scaler.transform(test_df.to_numpy())

    return X_train, y_train, X_test, y_test, X_test_sentences, train_df.columns

mlp_clf = MLPClassifier(verbose=True, hidden_layer_sizes=(20,), max_iter=50)
rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, verbose=0,
                                max_features='sqrt', max_leaf_nodes=25600)
lsvc_clf = LinearSVC(max_iter=1000, dual=False, penalty='l1', verbose=0)
lscv_clf_proba = CalibratedClassifierCV(lsvc_clf, n_jobs=-1)
voting_clf = VotingClassifier([('mlp', mlp_clf), ('rf', rf_clf), ('lsvc', CalibratedClassifierCV(lsvc_clf, n_jobs=-1))],
                              verbose=True, voting='soft', n_jobs=-1)

# TODO: save words used in bag of words
# TODO: coefficients from SVC
# TODO: test shuffled sentences against LLM and BOW
# TODO: Manually verify sentence embeddings with a few choice sentences

def feature_analysis():
    # Feature analysis for best features/model
    X_train, y_train, X_test, y_test, X_test_sentences, columns = load_features()
    columns = list(columns)

    deltas = []
    with open(base_dir + 'feature_importances.csv', 'w', newline='') as result_file:
        writer = csv.writer(result_file)

        prev_accuracy = None
        prev_column = None
        while X_train.shape[1] > 0:
            print('X_train dim', X_train.shape, X_train.shape[1])
            print('columns dim', len(columns))
            rf_clf.fit(X_train, y_train)
            feature_importances = list(zip(rf_clf.feature_importances_, columns))
            feature_importances.sort(reverse=True)
            print('Feature Importances', feature_importances)
            '''
            if prev_column is not None: writer.writerow([prev_column])
            writer.writerow([column for _, column in feature_importances])
            for importance, column in feature_importances:
                writer.writerow([column, importance])
            '''
            scores = cross_val_score(rf_clf, X_train, y_train, cv=10, n_jobs=1)
            accuracy_mean = scores.mean()
            accuracy_std = scores.std()
            y_pred = rf_clf.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_pred)
            print('Accuracy for Random Forest on Features', accuracy_mean)
            
            col_idx = np.argmin(rf_clf.feature_importances_)
            if prev_accuracy is not None:
                delta_accuracy = prev_accuracy - accuracy_mean
                print(prev_column, 'contributed', delta_accuracy)
                deltas.append([prev_column, accuracy_mean, accuracy_std,
                               delta_accuracy, accuracy_test])
            prev_accuracy = accuracy_mean
            prev_column = columns[col_idx]
            print('Removing ', col_idx, columns[col_idx])
            X_train = np.delete(X_train, col_idx, axis=1)
            X_test = np.delete(X_test, col_idx, axis=1)
            columns.pop(col_idx)
            #print('columns', columns)
        writer.writerow(['Feature', 'Accuracy Mean', 'Accuracy Std', 'Delta Accuracy', 'Test Accuracy'])
        for delta in deltas:
            writer.writerow(delta)

# Additive
def feature_analysis2():
    # Feature analysis for best features/model
    X_train, y_train, X_test, y_test, X_test_sentences, columns = load_features()
    columns = list(columns)

    deltas = []
    with open(base_dir + 'feature_importances2.csv', 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['Feature', 'Accuracy Mean', 'Accuracy Std', 'Delta Accuracy', 'Test Accuracy'])

        added_features = []
        remaining_features = {i for i in range(X_train.shape[1])}
        for j in range(10):
            next_best_feature = -1
            next_best_accuracy = 0
            for i in remaining_features:
                X_train_curr = X_train[:, added_features + [i]]
                rf_clf.fit(X_train_curr, y_train)
                y_pred = rf_clf.predict(X_test[:, added_features + [i]])
                accuracy_test = accuracy_score(y_test, y_pred)
                if next_best_feature == -1 or accuracy_test > next_best_accuracy:
                    next_best_feature = i
                    next_best_accuracy = accuracy_test
            
            added_features.append(next_best_feature)
            remaining_features.remove(next_best_feature)
            X_train_curr = X_train[:, added_features]
            scores = cross_val_score(rf_clf, X_train_curr, y_train, cv=10, n_jobs=1)
            accuracy_mean = scores.mean()
            accuracy_std = scores.std()
            row = [columns[next_best_feature], accuracy_mean, accuracy_std, next_best_accuracy]
            print(row)
            writer.writerow(row)

def content_vs_structure():
    train_df = load_df('training_data_raw.features.csv').fillna(0)
    y_train = train_df['label'].to_numpy().astype('int')
    test_df = load_df('test_data_raw.features.csv').fillna(0)
    y_test = test_df['label'].to_numpy().astype('int')
    
    structure_columns = ['word_count', 'num_count', 'punct_count', 'uniq_1_grams', 'uniq_2_grams', 'uniq_3_grams',
                         'Determiner', 'Verb', 'Pronoun', 'Preposition', 'Noun','Conjunction', 'Adjective', 'UnknownPoS',
                         'Article', 'ProperNoun', 'Adverb']
    content_columns = ['pct_simple', 'mean_syll', 'mean_pct_known', 'mean_aoa', 'max_aoa', 'mean_conc_mean',
                       'max_conc_mean', 'mean_conc_std', 'mean_freqs']

    with open(base_dir + 'content_vs_structure.csv', 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['Features', 'Mean Acc CV', 'Std Acc CV', 'Test Acc'])
        for columns, label in [(structure_columns, 'Structure Only'),
                               (content_columns, 'Content Only'),
                               (structure_columns + content_columns, 'Structure + Content')]:
            X_train = train_df[columns].to_numpy()
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = test_df[columns].to_numpy()
            X_test = scaler.transform(X_test)
            
            scores = cross_val_score(rf_clf, X_train, y_train, cv=5, n_jobs=1)
            rf_clf.fit(X_train, y_train)
            y_pred = rf_clf.predict(X_test)
            accuracy_test = accuracy_score(y_test, y_pred)
            row = [label, scores.mean(), scores.std(), accuracy_test]
            writer.writerow(row)
            print(row)
    

def bag_of_words_max_features():
    train_df = load_df('training_data_raw.csv')
    X_train_sentences = train_df['original_text'].str.lower().to_numpy()
    y_train = train_df['label'].to_numpy()
    print('y_train', y_train[:2])

    with open(base_dir + 'max_features_bag_of_words.csv', 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['Vocab Size', 'Mean Acc CV', 'Std Acc CV'])
        
        vocab_sizes = [100 * 2**i for i in range(11)]   
        for i in vocab_sizes:
            word_bagger = TfidfVectorizer(max_features=i)
            word_bagger.fit_transform(X_train_sentences)
            
            #print('number of unique words', i, len(word_bagger.vocabulary_))
        
            X_train = word_bagger.transform(X_train_sentences)
            y_train = y_train.astype('int')
            scores = cross_val_score(lsvc_clf, X_train, y_train, cv=num_folds, n_jobs=-1)
            print(len(word_bagger.vocabulary_), scores.mean(), scores.std())

            writer.writerow([len(word_bagger.vocabulary_), scores.mean(), scores.std()])

def hyper_tuning():
    rf_clf = RandomForestClassifier(n_jobs=32)
    X_train, y_train, X_test, y_test, X_test_sentences, columns  = load_features()
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    param_grid={'max_leaf_nodes': [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400],
                #'n_estimators': [10, 20, 40, 80, 160, 320],
                #'max_features': [1, 2, 4, 8, 16]
                }
    grid_clf = GridSearchCV(rf_clf, return_train_score=True, verbose=3, cv=5, param_grid=param_grid)
    grid_clf.fit(X_train, y_train)
    grid_df = pd.DataFrame(grid_clf.cv_results_)
    print(grid_df.head())
    grid_df.to_csv(base_dir + 'RandomForest.hypertuning.max_leaf_nodes.csv')

    param_grid={#'max_leaf_nodes': [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600],
                'n_estimators': [10, 20, 40, 80, 160, 320, 640, 1280, 2560],
                #'max_features': [1, 2, 4, 8, 16]
                }
    grid_clf = GridSearchCV(rf_clf, return_train_score=True, verbose=3, cv=5, param_grid=param_grid)
    grid_clf.fit(X_train, y_train)
    grid_df = pd.DataFrame(grid_clf.cv_results_)
    print(grid_df.head())
    grid_df.to_csv(base_dir + 'RandomForest.hypertuning.n_estimators.csv')

    param_grid={#'max_leaf_nodes': [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600],
                #'n_estimators': [10, 20, 40, 80, 160, 320],
                'max_features': [1, 2, 4, 8, 16]
                }
    grid_clf = GridSearchCV(rf_clf, return_train_score=True, verbose=3, cv=5, param_grid=param_grid)
    grid_clf.fit(X_train, y_train)
    grid_df = pd.DataFrame(grid_clf.cv_results_)
    print(grid_df.head())
    grid_df.to_csv(base_dir + 'RandomForest.hypertuning.max_features.csv')



def hyper_tuning2():
    svc_clf = SVC()
    X_train, y_train, X_test, columns = load_features_and_embeddings()
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=0.1, random_state=0)
    X_train, y_train = shuffle(X_train, y_train)
    X_validation, y_validation = shuffle(X_validation, y_validation)
    param_grid={'C': [0.1, 1, 10, 100, 1000],
                'gamma': [0.01, 0.1, 1, 10, 100]
                }
    grid_clf = GridSearchCV(svc_clf, return_train_score=True, verbose=3, cv=3, param_grid=param_grid)
    grid_clf.fit(X_train, y_train)
    grid_df = pd.DataFrame(grid_clf.cv_results_)
    print(grid_df.head())
    grid_df.to_csv(base_dir + 'SVC.hypertuning.csv')

def save_predictions(X_test, y_test, X_test_sentences, X_predictions, clf, name):
    #clf.fit(X_test, y_test)
    y_pred = clf.predict(X_test)
    hold_out_accuracy = accuracy_score(y_test, y_pred)
    print('**************hold out accuracy = ', hold_out_accuracy)
    y_pred_prob = clf.predict_proba(X_test)

    for i in range(X_test.shape[0]):
        sentence = X_test_sentences[i]
        if sentence not in X_predictions:
            X_predictions[sentence] = {}
            X_predictions[sentence]['label'] = y_test[i]
        # likelihood that sentence is complex. threshold is implied to be 0.5
        X_predictions[sentence][name] = y_pred_prob[i][1] #y_pred[i]
        #X_predictions[sentence][name + ' prob'] = y_pred_prob[i][1]
    
    return hold_out_accuracy

def run_models():
    results = []
    
    prediction_columns = []
    # sentence -> prediction
    X_predictions = {}
    for dataset, dataset_name in [#(load_bag_of_words_and_features, 'Merged'),
                                  #(load_embeddings, 'Word Embeddings'),
                                  (load_features, 'Engineered Features'),
                                  #(load_features_and_embeddings, 'Features + Embeddings'),
                                  (load_bag_of_words, 'Bag of Words'),
                                  ]:
        X_train, y_train, X_test, y_test, X_test_sentences, columns = dataset()
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                        test_size=0.1, random_state=0)
        #print(dataset_name, 'X:', X_train[:10])
        #print(dataset_name, 'y:', y_train[:10])
        for clf, model_name, n_jobs in [#(voting_clf, 'VotingClassifier'),
                                (lscv_clf_proba, 'LinearSVC', -1),
                                (rf_clf, 'RandomForest', 1),
                                (mlp_clf, 'MLP', 3)
                                ]:
            scores = cross_val_score(clf, X_train, y_train, cv=num_folds, n_jobs=n_jobs)
            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_validation)
            #accuracy = accuracy_score(y_validation, y_pred)
            
            ### Save predictions
            clf.fit(X_train, y_train)
            test_acc = save_predictions(X_test, y_test, X_test_sentences, X_predictions,
                                        clf, dataset_name + '/' + model_name)
    
            results.append((dataset_name, model_name, scores.mean(), scores.std(), test_acc))
            print('Accuracy/Std for', dataset_name, model_name, scores.mean(), scores.std(), test_acc)

    with open(base_dir + 'supervised_results.csv', 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['Dataset', 'Model', 'Mean Acc CV', 'Std Acc CV', 'Test Acc'])
        for dataset_name, model_name, accuracy_mean, accuracy_std, test_acc in results:
            writer.writerow([dataset_name, model_name, accuracy_mean, accuracy_std, test_acc])
    
    with open(base_dir + 'supervised_predictions.csv', 'w', newline='') as prediction_file:
        writer = csv.writer(prediction_file)
        keys = None
        for sentence, predictions in X_predictions.items():
            if keys is None:
                # enforce same key order for all rows
                keys = list(predictions.keys())
                # write header
                writer.writerow(['sentence'] + keys)
            row = [sentence]
            for key in keys:
                row.append(predictions[key])
            writer.writerow(row)

    '''        
    pred_df = pd.DataFrame()
    for sentence, predictions in X_predictions.items():
        predictions['sentence'] = sentence
        pred_df = pred_df.append(predictions, ignore_index=True)
    pred_df.to_csv(base_dir + 'supervised_predictions.csv')
    '''
    print(results)

content_vs_structure()
bag_of_words_max_features()
feature_analysis2()
hyper_tuning()
run_models()
