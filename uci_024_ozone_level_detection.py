#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data'

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
    'Date',
    'WSR0',
    'WSR1',
    'WSR2',
    'WSR3',
    'WSR4',
    'WSR5',
    'WSR6',
    'WSR7',
    'WSR8',
    'WSR9',
    'WSR10',
    'WSR11',
    'WSR12',
    'WSR13',
    'WSR14',
    'WSR15',
    'WSR16',
    'WSR17',
    'WSR18',
    'WSR19',
    'WSR20',
    'WSR21',
    'WSR22',
    'WSR23',
    'WSR_PK',
    'WSR_AV',
    'T0',
    'T1',
    'T2',
    'T3',
    'T4',
    'T5',
    'T6',
    'T7',
    'T8',
    'T9',
    'T10',
    'T11',
    'T12',
    'T13',
    'T14',
    'T15',
    'T16',
    'T17',
    'T18',
    'T19',
    'T20',
    'T21',
    'T22',
    'T23',
    'T_PK',
    'T_AV',
    'T85',
    'RH85',
    'U85',
    'V85',
    'HT85',
    'T70',
    'RH70',
    'U70',
    'V70',
    'HT70',
    'T50',
    'RH50',
    'U50',
    'V50',
    'HT50',
    'KI',
    'TT',
    'SLP',
    'SLP_',
    'Precp',
    'target']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, header = None, names = columns, index_col = False, na_values = '?')

# drop Date because it's not a feature for modeling
df_train = df_train.drop('Date', axis = 1)

# the target_ozone_day variable, inserted as first column, and drop the original target column
df_train.insert(0, 'target_ozone_day', df_train['target'])
df_train = df_train.drop('target', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_024_ozone_Level_detection.csv', index = False)
