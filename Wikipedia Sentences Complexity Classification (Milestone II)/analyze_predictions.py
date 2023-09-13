#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:44:40 2023

@author: thanuja
"""

base_dir = '/home/thanuja/Dropbox/coursera/Milestone2/data/'

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

#plt.rcParams['figure.figsize'] = [4, 4]

legend_font_size=10
label_font_size=12

columns = ['Bag of Words/LinearSVC', 'Bag of Words/RandomForest', 'Bag of Words/MLP',
           #'Features + Embeddings/LinearSVC', 'Features + Embeddings/RandomForest', 'Features + Embeddings/MLP',
           #'Word Embeddings/LinearSVC', 'Word Embeddings/RandomForest', 'Word Embeddings/MLP',
           'Engineered Features/LinearSVC', 'Engineered Features/RandomForest',
           'Engineered Features/MLP', 'GRU on PoS', 'Distilbert']
col_abbrs = ['BoW/SVC', 'BoW/RF', 'BoW/MLP',
           #'Feat+Emb/SVC', 'Feat+Emb/RF', 'Feat+Emb/MLP',
           #'Embed/SVC', 'Embed/RF', 'Embed/MLP',
           'Features/SVC', 'Features/RF', 'Features/MLP',
           'GRU on PoS', 'Distilbert', 'Label']
col_abbr_map = {}
for i in range(len(columns)):
    col_abbr_map[columns[i]] = col_abbrs[i]
col_abbr_map['Raw text/LLM'] = 'Raw Text/LLM'
col_abbr_map['PoS Sequences/GRU'] = 'PoS Seq/GRU'
col_abbr_map['label'] = 'label'

### OVERALL RESULTS ###

overall_results_df = pd.read_csv(base_dir + 'supervised_results.csv')
overall_results_df['label'] = overall_results_df['Dataset'] + '/' + overall_results_df['Model']
overall_results_df['label'] = overall_results_df['label'].map(col_abbr_map)
overall_results_df['Mean Acc CV'] = overall_results_df['Mean Acc CV'].round(3)

plt.clf()
plt.figure(figsize=(5,5))
ax = overall_results_df.plot.barh(x='label', y='Mean Acc CV', xlim=[0.5, 0.8],
                        xerr='Std Acc CV', title='Cross Validated Accuracy', legend=None,
                        xlabel=None, ylabel=None)
ax.bar_label(ax.containers[1])
#plt.axes().get_yaxis().set_visible(False)
plt.subplots_adjust(left=0.18)
plt.savefig(base_dir + 'supervised_results.png')
plt.tight_layout()
plt.close()

### CONTENT VS STRUCTURE ###

feature_results_df = pd.read_csv(base_dir + 'content_vs_structure.csv')
feature_results_df['Mean Acc CV'] = feature_results_df['Mean Acc CV'].round(3)
plt.clf()
plt.figure(figsize=(4,4))
ax = feature_results_df.plot.barh(x='Model + Features', y='Mean Acc CV', xlim=[0.6, 0.73],
                        xerr='Std Acc CV', title='Cross Validated Accuracy', legend=None,
                        xlabel=None, ylabel=None)
ax.bar_label(ax.containers[1])
#plt.axes().get_yaxis().set_visible(False)
plt.subplots_adjust(left=0.3)
plt.savefig(base_dir + 'content_vs_structure.png')
plt.tight_layout()
plt.close()


### FEATURE IMPORTANCES ### 

feature_importances_df = pd.read_csv(base_dir + 'feature_importances2.csv').iloc[::-1]
feature_importances_df['Accuracy Mean'] = feature_importances_df['Accuracy Mean'].round(3)
feature_importances_df['Delta Accuracy'] = feature_importances_df['Delta Accuracy'].round(5)

plt.clf()
plt.figure(figsize=(7, 4))
ax = feature_importances_df.plot.barh(x='Feature', y='Accuracy Mean', xlim=[0.5, 0.75],
                        xerr='Accuracy Std', title='Feature Importances', legend=None)
plt.xlabel('Accuracy %')
ax.bar_label(ax.containers[1])
#plt.axes().get_yaxis().set_visible(False)
plt.subplots_adjust(left=0.23)
plt.savefig(base_dir + 'feature_importances2.png')
#plt.tight_layout()
plt.close()


### VOCAB SIZE VS ACCURACY ###

max_results_df = pd.read_csv(base_dir + 'max_features_bag_of_words.csv')
plt.clf()
plt.figure(figsize=(4, 3))
max_results_df.plot.line(x='Vocab Size', y='Mean Acc CV', yerr='Std Acc CV', title='Num Words on BoW SVC Accuracy',
                         legend=None, style='o-')
plt.xscale('log')
plt.xlabel('Number of Words')
plt.ylabel('LinearSVC Accuracy')
plt.tight_layout()
plt.savefig(base_dir + 'max_features_bag_of_words.png')
plt.close()


### RANDOM FOREST HYPERTUNING ###

def plot_hypertuning(param, add_train):
    rf_hypertuning_df = pd.read_csv(base_dir + 'RandomForest.hypertuning.' + param + '.csv')
    plt.clf()
    plt.figure(figsize=(4, 4))
    plt.title('Random Forest ' + param)
    x = rf_hypertuning_df['param_' + param]
    plt.errorbar(x, rf_hypertuning_df.mean_test_score, yerr=rf_hypertuning_df.std_test_score,
                 label='Validation', fmt='o-')
    if add_train:
        plt.errorbar(x, rf_hypertuning_df.mean_train_score, yerr=rf_hypertuning_df.std_train_score,
                     label='Train', fmt='o-')
    plt.xlabel(param)
    plt.ylabel('Random Forest Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir + 'RandomForest.hypertuning.' + param + ('.train' if add_train else '') + '.png')
    plt.close()


for param in ['max_features', 'n_estimators', 'max_leaf_nodes']:
    plot_hypertuning(param, True)
    plot_hypertuning(param, False)


### LLM LEARNING CURVE ###

llm_learning_df = pd.read_csv(base_dir + 'llm_learning_curve.csv')
gru_learning_df = pd.read_csv(base_dir + 'gru_learning_curve.csv')

plt.clf()
plt.figure(figsize=(5, 3))
plt.title('Dataset Size on Accuracy')
plt.plot(llm_learning_df['Num Sentences'], llm_learning_df['Accuracy'], marker='o', label='LLM Distilbert')
plt.plot(gru_learning_df['Num Sentences'], gru_learning_df['Accuracy'], marker='o', label='GRU on PoS')
#llm_learning_df.plot.line(x='Num Sentences', y='Accuracy', legend=None, style='o-', label='LLM Distilbert')
#gru_learning_df.plot.line(x='Num Sentences', y='Accuracy', legend=None, style='o-', label='GRU on PoS')
plt.xscale('log')
plt.xlabel('Number of Sentences')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(base_dir + 'llm_gru_learning_curve.png')
plt.close()


### PREDICTION CORRELATIONS ###

notable_columns = ['Bag of Words/LinearSVC', 'Engineered Features/RandomForest',
           'GRU on PoS', 'Distilbert']
notable_abbrs = [col_abbr_map[col] for col in notable_columns]

predictions_df = pd.read_csv(base_dir + 'supervised_predictions.csv')
labels = predictions_df['label'].to_numpy()

corr_df = predictions_df[notable_columns + ['label']]
corr = corr_df.corr()

plt.figure(figsize=(6,6))
sns.heatmap(corr, xticklabels=notable_abbrs + ['label'],
            yticklabels=notable_abbrs + ['label'], annot=True, cmap="crest")
plt.subplots_adjust(bottom=0.2, left=0.05, right=1.2, top=0.95)
plt.savefig(base_dir + 'prediction_correlations.png')
plt.tight_layout()
plt.close()


### ROC AUC curve ###
# from https://stackoverflow.com/a/38467407
plt.clf()
plt.figure(figsize=(4,4))
plt.title('ROC Curve')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('FPR', fontsize=label_font_size)
plt.ylabel('TPR', fontsize=label_font_size)
for i in range(len(notable_columns)):
    col = notable_columns[i]
    pred_probs = predictions_df[col].to_numpy()
    fpr, tpr, thresholds = metrics.roc_curve(labels, pred_probs)
    roc_auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(labels, pred_probs.round())
    plt.plot(fpr, tpr, label = notable_abbrs[i] + ', AUC = %0.2f' % (roc_auc))
plt.legend(loc = 'lower right', fontsize=legend_font_size)
plt.tight_layout()
plt.savefig(base_dir + 'roc_curve.png')
plt.close()


### PRECISION RECALL CURVE ###
plt.clf()
plt.figure(figsize=(4,4))
plt.title('Precision Recall')
plt.xlabel('Recall', fontsize=label_font_size)
plt.ylabel('Precision', fontsize=label_font_size)
for i in range(len(notable_columns)):
    col = notable_columns[i]
    pred_probs = predictions_df[col].to_numpy()
    precision, recall, thresholds = metrics.precision_recall_curve(labels, pred_probs)
    auc_score = metrics.auc(recall, precision)
    accuracy = metrics.accuracy_score(labels, pred_probs.round())
    plt.plot(recall, precision, label = notable_abbrs[i] + ', AUC = %0.2f' % (auc_score))
plt.legend(loc = 'lower left', fontsize=legend_font_size)
plt.tight_layout()
plt.savefig(base_dir + 'precision_recall_curve.png')
plt.close()


### SIGNIFICANT EXAMPLE PREDICTIONS

pred_cols = predictions_df[notable_columns].to_numpy()
means = pred_cols.mean(axis=1)
stds = pred_cols.std(axis=1)
predictions_df['probability_mean'] = means
predictions_df['probability_std'] = stds
predictions_df['abs_error_mean'] = np.abs(means - labels)
predictions_df['range'] = pred_cols.max(axis=1) - pred_cols.min(axis=1)
predictions_df['weighted_prob'] = (predictions_df['Bag of Words/LinearSVC'] \
    + predictions_df['GRU on PoS'] + predictions_df['Engineered Features/RandomForest'] \
    + predictions_df['Distilbert']) / 4
predictions_df['weighted_prob_abs_error'] = np.abs(predictions_df['weighted_prob'].to_numpy() - labels)

for col in columns:
    pred_probs = predictions_df[col].to_numpy()
    col_err = np.abs(pred_probs - labels)
    predictions_df['Error for ' + col] = col_err

predictions_df.reset_index(inplace=True, drop=True)


### ERROR CORRELATIONS ###

plt.figure(figsize=(6,6))
corr_df = predictions_df[['Error for ' + col for col in notable_columns] + ['label']]
corr = corr_df.corr()
sns.heatmap(corr, xticklabels=notable_abbrs + ['label'],
            yticklabels=notable_abbrs + ['label'], annot=True, cmap="crest")
plt.subplots_adjust(bottom=0.2, left=0.05, right=1.2, top=0.95)
plt.tight_layout()
plt.savefig(base_dir + 'prediction_error_correlations.png')
plt.close()

## Drop unwanted 'Unnamed: 0.1.1' style columns that keep getting created on every save.
to_drop = []
for col in predictions_df.columns:
    if col.startswith('Unnamed:'): to_drop.append(col)
predictions_df.drop(columns=to_drop, inplace=True)

## Save updated CSV
predictions_df.to_csv(base_dir + 'supervised_predictions.csv')
