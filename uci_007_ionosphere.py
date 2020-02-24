#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Ionosphere

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'

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

# target is the original target variable, which will be converted into 0 or 1 later
columns = [
    'variable_01',
    'variable_02',
    'variable_03',
    'variable_04',
    'variable_05',
    'variable_06',
    'variable_07',
    'variable_08',
    'variable_09',
    'variable_10',
    'variable_11',
    'variable_12',
    'variable_13',
    'variable_14',
    'variable_15',
    'variable_16',
    'variable_17',
    'variable_18',
    'variable_19',
    'variable_20',
    'variable_21',
    'variable_22',
    'variable_23',
    'variable_24',
    'variable_25',
    'variable_26',
    'variable_27',
    'variable_28',
    'variable_29',
    'variable_30',
    'variable_31',
    'variable_32',
    'variable_33',
    'variable_34',
    'target']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, header = None, names = columns, index_col = False)

# the target variable, 0 = b (bad) and 1 = g (good)
df_train['target'] = df_train['target'].apply(lambda x: 0 if x == 'b' else 1)

# re-order the columns so that target becomes the first column
df_train = df_train[[
    'target',
    'variable_01',
    'variable_02',
    'variable_03',
    'variable_04',
    'variable_05',
    'variable_06',
    'variable_07',
    'variable_08',
    'variable_09',
    'variable_10',
    'variable_11',
    'variable_12',
    'variable_13',
    'variable_14',
    'variable_15',
    'variable_16',
    'variable_17',
    'variable_18',
    'variable_19',
    'variable_20',
    'variable_21',
    'variable_22',
    'variable_23',
    'variable_24',
    'variable_25',
    'variable_26',
    'variable_27',
    'variable_28',
    'variable_29',
    'variable_30',
    'variable_31',
    'variable_32',
    'variable_33',
    'variable_34']]

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_007_ionosphere.csv', index = False)
