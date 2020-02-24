#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import numpy
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'

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

# unzip the downloaded file, get df_train and df_test
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('bank-full.csv') as myfile:
        df_train = pandas.read_csv(io.BytesIO(myfile.read()), delimiter = ';')

    with myzip.open('bank.csv') as myfile:
        df_test = pandas.read_csv(io.BytesIO(myfile.read()), delimiter = ';')

# 2 dataframes will be concatenated for data processing, a "train_or_test" label makes it easier for separating them later
df_train['train_or_test'] = 'train'
df_test['train_or_test'] = 'test'
df_total = pandas.concat([df_train, df_test]).reset_index(drop = True)

# drop day and month, because they are not appropriate for modeling
df_total = df_total.drop(['day', 'month'], axis = 1)

# set pdays = -1 to missing, avoid binning -1 with other scalar values when modeling
df_total['pdays'] = df_total['pdays'].apply(lambda x: numpy.nan if x == -1 else x)

# the target variable, y = 1 (yes) and y = 0 (no)
df_total['target_y'] = df_total['y'].apply(lambda x: 1 if x == 'yes' else 0)
df_total = df_total.drop('y', axis = 1)

# re-order the columns so that target_y becomes the first column
df_total = df_total[[
    'target_y',
    'age',
    'marital',
    'education',
    'default',
    'balance',
    'housing',
    'loan',
    'contact',
    'duration',
    'campaign',
    'pdays',
    'previous',
    'poutcome',
    'train_or_test',
    'job']]

# separate train and test data samples, and drop the train_or_test label
bank_marketing_train = df_total[df_total['train_or_test'] == 'train'].drop('train_or_test', axis = 1)
bank_marketing_test = df_total[df_total['train_or_test'] == 'test'].drop('train_or_test', axis = 1)

# save the 2 dataframes as CSV files, you can zip them respectively, upload them to t1modeler.com, and build a model
bank_marketing_train.to_csv('uci_002_bank_marketing_train.csv', index = False)
bank_marketing_test.to_csv('uci_002_bank_marketing_test.csv', index = False)
