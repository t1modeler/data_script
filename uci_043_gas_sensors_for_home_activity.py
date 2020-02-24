#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Gas+sensors+for+home+activity+monitoring

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00362/HT_Sensor_UCIsubmission.zip'

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
    with myzip.open('HT_Sensor_metadata.dat') as myfile:
        df_metadata = pandas.read_csv(myfile, delimiter = '\s+', header = 0, index_col = None)

    with myzip.open('HT_Sensor_dataset.zip') as myfile:
        with zipfile.ZipFile(myfile) as inner_zip:
            with inner_zip.open('HT_Sensor_dataset.dat') as inner_file:
                df_sensor = pandas.read_csv(inner_file, delimiter = '\s+', header = 0, index_col = None)

# merge class with sensor data
df_total = df_sensor.merge(df_metadata[['id', 'class']], how = 'left', on = 'id')

# drop variables which are not for modeling
df_total = df_total.drop(['id', 'time'], axis = 1)

# the target variable, inserted into the dataframe as the first column, and drop the original class variable
# define class = wine as 1 and other values as 0
df_total.insert(0, 'target_class', df_total['class'].apply(lambda x: 1 if x == 'wine' else 0))
df_total = df_total.drop('class', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_total.to_csv('uci_043_gas_sensors_for_home_activity.csv', index = False)
