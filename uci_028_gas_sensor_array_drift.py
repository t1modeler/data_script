#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import sklearn.datasets
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip'

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

# generate column names
base_name = ['IR_', '|IR|_', 'EMAi0.001_', 'EMAi0.01_', 'EMAi0.1_', 'EMAd0.001_', 'EMAd0.01_', 'EMAd0.1_']
feature_columns = []
for i in range(1, 17):
    suffix_added = [j + str(i) for j in base_name]
    feature_columns.extend(suffix_added)

# unzip the downloaded file, and get data files
df_train = pandas.DataFrame()
with zipfile.ZipFile(data_train) as myzip:
    file_names = ['Dataset/batch{0}.dat'.format(i) for i in range(1, 11)]
    for file in file_names:
        with myzip.open(file) as myfile:
            df_x, df_y = sklearn.datasets.load_svmlight_file(myfile)
            df_x = pandas.DataFrame(df_x.todense(), columns = feature_columns)
            df_y = pandas.DataFrame(df_y, columns = ['target_gas_substance'])
            df_one_file = df_y.merge(df_x, how = 'left', left_index = True, right_index = True)
            df_train = pandas.concat([df_train, df_one_file])

# the target variable, we binarize it as 1 = 1 (Ethanol) and 0 = other values
df_train['target_gas_substance'] = df_train['target_gas_substance'].apply(lambda x: 1 if x == 1 else 0)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_028_gas_sensor_array_drift.csv', index = False)
