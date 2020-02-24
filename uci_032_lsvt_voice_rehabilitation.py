#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00282/LSVT_voice_rehabilitation.zip'

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

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('LSVT_voice_rehabilitation.xlsx') as myfile:
        df_binary_response = pandas.read_excel(myfile, sheet_name = 'Binary response')
        df_binary_response = df_binary_response.rename(columns = {'Binary class 1=acceptable, 2=unacceptable': 'target_class'})

    with myzip.open('LSVT_voice_rehabilitation.xlsx') as myfile:
        df_subject_demographics = pandas.read_excel(myfile, sheet_name = 'Subject demographics')
        df_subject_demographics = df_subject_demographics.rename(columns = {'Gender, 0->Male, 1->Female': 'Gender'})
        df_subject_demographics = df_subject_demographics.drop('Subject_index', axis = 1)

    with myzip.open('LSVT_voice_rehabilitation.xlsx') as myfile:
        df_data = pandas.read_excel(myfile, sheet_name = 'Data')

# merge all dataframes
df_train = df_binary_response.merge(df_subject_demographics, how = 'left', left_index = True, right_index = True)
df_train = df_train.merge(df_data, how = 'left', left_index = True, right_index = True)

# set target variable to 1 (acceptable) and 0 (unacceptable)
df_train['target_class'] = df_train['target_class'].apply(lambda x: 1 if x == 1 else 0)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_032_lsvt_voice_rehabilitation.csv', index = False)
