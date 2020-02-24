#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/vlbw.html

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/vlbw.zip'

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

# download data from website
data_train = download_file(url_data_train) if url_data_train.startswith('http') else url_data_train

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('vlbw.csv') as myfile:
        df_train = pandas.read_csv(myfile, header = 0, index_col = 0)

# drop records with missing sex values, becuase they have very high missing rate for other variables
df_train = df_train[df_train['sex'].notnull()]

# the target variable, inserted into the dataframe as the first column, and drop the original dead variable
df_train.insert(0, 'target_dead', df_train['dead'])
df_train = df_train.drop('dead', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('vanderbilt_008_very_low_birth_weight_infants.csv', index = False)
