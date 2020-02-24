#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Adult

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
url_data_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

def download_file(url):
    resp = urllib.request.urlopen(url)
    if resp.status != 200:
        resp.close()
        raise ValueError('Error: {0}'.format(resp.reason))

    print('\rStarted', end = '\r')
    content_length = resp.getheader('Content-Length')
    if content_length is None:
        content_length = '(total: unknown)'
    else:
        content_length = int(content_length)
        if content_length < 1024:
            content_length_str = '(total %.0f Bytes)' % content_length
        elif content_length < 1024 * 1024:
            content_length_str = '(total %.0f KB)' % (content_length / 1024)
        else:
            content_length_str = '(total %.1f MB)' % (content_length / 1024 / 1024)

    total = bytes()
    while not resp.isclosed():
        total += resp.read(10 * 1024)
        if len(total) < 1024:
            print(('\rDownloaded: %.0f Bytes ' % len(total)) + content_length_str + '  ', end = '\r')
        if len(total) < 1024 * 1024:
            print(('\rDownloaded: %.0f KB ' % (len(total) / 1024)) + content_length_str + '  ', end = '\r')
        else:
            print(('\rDownloaded: %.1f MB ' % (len(total) / 1024 / 1024)) + content_length_str + '  ', end = '\r')

    print()
    return io.BytesIO(total)

# download data from UCI Machine Learning Repository
data_train = download_file(url_data_train) if url_data_train.startswith('http') else url_data_train
data_test = download_file(url_data_test) if url_data_test.startswith('http') else url_data_test

# income_greater_than_50k is the original target variable, which will be converted into 0 or 1 later
columns = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income_greater_than_50k']

df_train = pandas.read_csv(data_train, header = None, names = columns, index_col = False, skipinitialspace = True)
# "skiprows = 1" skip the first row in data_test, because it is not a data row
df_test = pandas.read_csv(data_test, header = None, names = columns, index_col = False, skipinitialspace = True, skiprows = 1)

# 2 dataframes will be concatenated for data processing, a "train_or_test" label makes it easier for separating them later
df_train['train_or_test'] = 'train'
df_test['train_or_test'] = 'test'
df_total = pandas.concat([df_train, df_test]).reset_index(drop = True)

# drop fnlwgt because it is not a variable for modeling
# drop education because education-num is a better variable for modeling
# drop relationship because it is not a variable for modeling
df_total = df_total.drop(['fnlwgt', 'education', 'relationship'], axis = 1)

# group fragmented workclass values into general categories
df_total['workclass'] = df_total['workclass'].apply(lambda x:
         'Self'    if x in ['Self-emp-not-inc', 'Self-emp-inc']
    else 'Gov'     if x in ['Local-gov', 'State-gov', 'Federal-gov']
    else 'Unknown' if x in ['?', 'Without-pay', 'Never-worked']
    else x)

# group fragmented marital-status values into general categories
df_total['marital-status'] = df_total['marital-status'].apply(lambda x:
         'Married'  if x in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse', 'Widowed']
    else 'Divorced' if x in ['Divorced', 'Separated']
    else x)

# group fragmented race values into general categories
df_total['race'] = df_total['race'].apply(lambda x:
    'Asian_AmerIndian_Other' if x in ['Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    else x)

# finally the target variable, we convert it into 0 (<=50K) or 1 (>50K)
df_total['target_income_greater_than_50k'] = df_total['income_greater_than_50k'].apply(lambda x: 1 if x == '>50K' else 0)
df_total = df_total.drop('income_greater_than_50k', axis = 1)

# re-order the columns so that target_income_greater_than_50k becomes the first column
df_total = df_total[[
    'target_income_greater_than_50k',
    'age',
    'workclass',
    'education-num',
    'marital-status',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'train_or_test',
    'native-country',
    'occupation']]

# separate train and test data samples, and drop the train_or_test label
census_income_train = df_total[df_total['train_or_test'] == 'train'].drop('train_or_test', axis = 1)
census_income_test = df_total[df_total['train_or_test'] == 'test'].drop('train_or_test', axis = 1)

# save the 2 dataframes as CSV files, you can zip them respectively, upload them to t1modeler.com, and build a model
census_income_train.to_csv('uci_001_adult_data_set_train.csv', index = False)
census_income_test.to_csv('uci_001_adult_data_set_test.csv', index = False)
