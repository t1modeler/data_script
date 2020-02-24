#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import numpy
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'

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

# convert Excel file into pandas dataframe
df_train = pandas.read_excel(data_train, header = 1, index_col = False)

# drop column ID
df_train = df_train.drop('ID', axis = 1)

# create feature sets - util (utilization rate, we prefer integers so util is multiplied by 100 and rounded)
df_train['util_1'] = numpy.round(df_train['BILL_AMT1'] / df_train['LIMIT_BAL'] * 100)
df_train['util_2'] = numpy.round(df_train['BILL_AMT2'] / df_train['LIMIT_BAL'] * 100)
df_train['util_3'] = numpy.round(df_train['BILL_AMT3'] / df_train['LIMIT_BAL'] * 100)
df_train['util_4'] = numpy.round(df_train['BILL_AMT4'] / df_train['LIMIT_BAL'] * 100)
df_train['util_5'] = numpy.round(df_train['BILL_AMT5'] / df_train['LIMIT_BAL'] * 100)
df_train['util_6'] = numpy.round(df_train['BILL_AMT6'] / df_train['LIMIT_BAL'] * 100)

# create aggregation features for util and drop the orignal util columns
df_train['util_recent_1'] = df_train['util_1']

df_train['util_recent_3_max'] = df_train[['util_1', 'util_2', 'util_3']].max(axis = 1)
df_train['util_recent_3_min'] = df_train[['util_1', 'util_2', 'util_3']].min(axis = 1)
df_train['util_recent_3_avg'] = df_train[['util_1', 'util_2', 'util_3']].mean(axis = 1)
df_train['util_recent_3_std'] = df_train[['util_1', 'util_2', 'util_3']].std(axis = 1)

df_train['util_recent_6_max'] = df_train[['util_1', 'util_2', 'util_3', 'util_4', 'util_5', 'util_6']].max(axis = 1)
df_train['util_recent_6_min'] = df_train[['util_1', 'util_2', 'util_3', 'util_4', 'util_5', 'util_6']].min(axis = 1)
df_train['util_recent_6_avg'] = df_train[['util_1', 'util_2', 'util_3', 'util_4', 'util_5', 'util_6']].mean(axis = 1)
df_train['util_recent_6_std'] = df_train[['util_1', 'util_2', 'util_3', 'util_4', 'util_5', 'util_6']].std(axis = 1)

df_train = df_train.drop(['util_1', 'util_2', 'util_3', 'util_4', 'util_5', 'util_6'], axis = 1)

# create aggregation features for PAY and drop the orignal PAY columns
df_train['PAY_recent_1'] = df_train['PAY_0']

df_train['PAY_recent_3_cnt_overdue'] = df_train[['PAY_0', 'PAY_2', 'PAY_3']].applymap(
    lambda x: 1 if x >= 1 else 0).sum(axis = 1)
df_train['PAY_recent_3_max'] = df_train[['PAY_0', 'PAY_2', 'PAY_3']].max(axis = 1)
df_train['PAY_recent_3_min'] = df_train[['PAY_0', 'PAY_2', 'PAY_3']].min(axis = 1)
df_train['PAY_recent_3_avg'] = df_train[['PAY_0', 'PAY_2', 'PAY_3']].mean(axis = 1)
df_train['PAY_recent_3_std'] = df_train[['PAY_0', 'PAY_2', 'PAY_3']].std(axis = 1)

df_train['PAY_recent_6_cnt_overdue'] = df_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].applymap(
    lambda x: 1 if x >= 1 else 0).sum(axis = 1)
df_train['PAY_recent_6_max'] = df_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].max(axis = 1)
df_train['PAY_recent_6_min'] = df_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].min(axis = 1)
df_train['PAY_recent_6_avg'] = df_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis = 1)
df_train['PAY_recent_6_std'] = df_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].std(axis = 1)

df_train = df_train.drop(['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], axis = 1)

# create aggregation features for BILL_AMT and drop the orignal BILL_AMT columns
df_train['BILL_AMT_recent_1'] = df_train['BILL_AMT1']

df_train['BILL_AMT_recent_3_cnt_zero'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']].applymap(
    lambda x: 1 if x == 0 else 0).sum(axis = 1)
df_train['BILL_AMT_recent_3_max'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']].max(axis = 1)
df_train['BILL_AMT_recent_3_min'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']].min(axis = 1)
df_train['BILL_AMT_recent_3_avg'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']].mean(axis = 1)
df_train['BILL_AMT_recent_3_std'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3']].std(axis = 1)

df_train['BILL_AMT_recent_6_cnt_zero'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].applymap(
    lambda x: 1 if x == 0 else 0).sum(axis = 1)
df_train['BILL_AMT_recent_6_max'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].max(axis = 1)
df_train['BILL_AMT_recent_6_min'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].min(axis = 1)
df_train['BILL_AMT_recent_6_avg'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis = 1)
df_train['BILL_AMT_recent_6_std'] = df_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].std(axis = 1)

df_train = df_train.drop(['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'], axis = 1)

# create aggregation features for PAY_AMT and drop the orignal PAY_AMT columns
df_train['PAY_AMT_recent_1'] = df_train['PAY_AMT1']

df_train['PAY_AMT_recent_3_cnt_zero'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']].applymap(
    lambda x: 1 if x == 0 else 0).sum(axis = 1)
df_train['PAY_AMT_recent_3_max'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']].max(axis = 1)
df_train['PAY_AMT_recent_3_min'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']].min(axis = 1)
df_train['PAY_AMT_recent_3_avg'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']].mean(axis = 1)
df_train['PAY_AMT_recent_3_std'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3']].std(axis = 1)

df_train['PAY_AMT_recent_6_cnt_zero'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].applymap(
    lambda x: 1 if x == 0 else 0).sum(axis = 1)
df_train['PAY_AMT_recent_6_max'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].max(axis = 1)
df_train['PAY_AMT_recent_6_min'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].min(axis = 1)
df_train['PAY_AMT_recent_6_avg'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis = 1)
df_train['PAY_AMT_recent_6_std'] = df_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].std(axis = 1)

df_train = df_train.drop(['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], axis = 1)

# the target variable, inserted as first column
df_train.insert(0, 'target_default_payment_next_month', df_train['default payment next month'].apply(lambda x: 1 if x == 1 else 0))
df_train = df_train.drop('default payment next month', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_019_default_of_credit_card_clients.csv', index = False)
