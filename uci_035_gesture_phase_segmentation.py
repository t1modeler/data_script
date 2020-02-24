#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Gesture+Phase+Segmentation

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00302/gesture_phase_dataset.zip'

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
    with myzip.open('a1_raw.csv') as myfile: df_a1_raw = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('a1_va3.csv') as myfile: df_a1_va3 = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('a2_raw.csv') as myfile: df_a2_raw = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('a2_va3.csv') as myfile: df_a2_va3 = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('a3_raw.csv') as myfile: df_a3_raw = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('a3_va3.csv') as myfile: df_a3_va3 = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('b1_raw.csv') as myfile: df_b1_raw = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('b1_va3.csv') as myfile: df_b1_va3 = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('b3_raw.csv') as myfile: df_b3_raw = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('b3_va3.csv') as myfile: df_b3_va3 = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('c1_raw.csv') as myfile: df_c1_raw = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('c1_va3.csv') as myfile: df_c1_va3 = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('c3_raw.csv') as myfile: df_c3_raw = pandas.read_csv(myfile, header = 0, index_col = False)
    with myzip.open('c3_va3.csv') as myfile: df_c3_va3 = pandas.read_csv(myfile, header = 0, index_col = False)

# merge a1, a2, a3, b1, b3, c1, c3 dataframes respectively
df_a1_total = df_a1_raw.merge(df_a1_va3, how = 'left', left_index = True, right_index = True)
df_a2_total = df_a2_raw.merge(df_a2_va3, how = 'left', left_index = True, right_index = True)
df_a3_total = df_a3_raw.merge(df_a3_va3, how = 'left', left_index = True, right_index = True)
df_b1_total = df_b1_raw.merge(df_b1_va3, how = 'left', left_index = True, right_index = True)
df_b3_total = df_b3_raw.merge(df_b3_va3, how = 'left', left_index = True, right_index = True)
df_c1_total = df_c1_raw.merge(df_c1_va3, how = 'left', left_index = True, right_index = True)
df_c3_total = df_c3_raw.merge(df_c3_va3, how = 'left', left_index = True, right_index = True)

# concatenate all dataframes
df_complete = pandas.concat([df_a1_total, df_a2_total, df_a3_total, df_b1_total, df_b3_total, df_c1_total, df_c3_total])

# rename the processed variables because their original names are confusing
df_complete = df_complete.rename(columns = {
    '1': 'processed_var_1',
    '2': 'processed_var_2',
    '3': 'processed_var_3',
    '4': 'processed_var_4',
    '5': 'processed_var_5',
    '6': 'processed_var_6',
    '7': 'processed_var_7',
    '8': 'processed_var_8',
    '9': 'processed_var_9',
    '10': 'processed_var_10',
    '11': 'processed_var_11',
    '12': 'processed_var_12',
    '13': 'processed_var_13',
    '14': 'processed_var_14',
    '15': 'processed_var_15',
    '16': 'processed_var_16',
    '17': 'processed_var_17',
    '18': 'processed_var_18',
    '19': 'processed_var_19',
    '20': 'processed_var_20',
    '21': 'processed_var_21',
    '22': 'processed_var_22',
    '23': 'processed_var_23',
    '24': 'processed_var_24',
    '25': 'processed_var_25',
    '26': 'processed_var_26',
    '27': 'processed_var_27',
    '28': 'processed_var_28',
    '29': 'processed_var_29',
    '30': 'processed_var_30',
    '31': 'processed_var_31',
    '32': 'processed_var_32'})

# drop variables not for modeling
df_complete = df_complete.drop(['timestamp', 'Phase'], axis = 1)

# the target variable, inserted into the dataframe as the first column, and drop the original phase variable
# see if we can distinguish Retraction from other gestures
df_complete.insert(0, 'target_phase', df_complete['phase'].apply(lambda x: 1 if x == 'Retraction' else 0))
df_complete = df_complete.drop('phase', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_complete.to_csv('uci_035_gesture_phase_segmentation.csv', index = False)
