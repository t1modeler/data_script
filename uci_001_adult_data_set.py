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

# drop fnlwgt because it is not a variable for modeling
# drop education because education-num is a better variable for modeling
# drop relationship because it is not a variable for modeling
df_train = df_train.drop(['fnlwgt', 'education', 'relationship'], axis = 1)

# group fragmented workclass values into general categories
df_train['workclass'] = df_train['workclass'].apply(lambda x:
         'Self'    if x in ['Self-emp-not-inc', 'Self-emp-inc']
    else 'Gov'     if x in ['Local-gov', 'State-gov', 'Federal-gov']
    else 'Unknown' if x in ['?', 'Without-pay', 'Never-worked']
    else x)

# group fragmented marital-status values into general categories
df_train['marital-status'] = df_train['marital-status'].apply(lambda x:
         'Married'  if x in ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse', 'Widowed']
    else 'Divorced' if x in ['Divorced', 'Separated']
    else x)

# group fragmented race values into general categories
df_train['race'] = df_train['race'].apply(lambda x:
    'Asian_AmerIndian_Other' if x in ['Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    else x)

# finally the target variable, we convert it into 0 (<=50K) or 1 (>50K)
df_train['target_income_greater_than_50k'] = df_train['income_greater_than_50k'].apply(lambda x: 1 if x == '>50K' else 0)
df_train = df_train.drop('income_greater_than_50k', axis = 1)

# re-order the columns so that target_income_greater_than_50k becomes the first column
df_train = df_train[[
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
    'native-country',
    'occupation']]

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_001_adult_data_set.csv', index = False)
