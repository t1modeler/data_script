#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import numpy
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt'

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
columns = ['variable_' + str(i + 1).zfill(3) for i in range(50)]

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, delimiter = '\s+', header = None, names = columns, index_col = False, skiprows = 1)

# convert -999 values into numpy.nan, because -999 values should not be binned with regular numeric values
for column in df_train.columns:
    df_train[column] = df_train[column].apply(lambda x: numpy.nan if x == -999 else x).astype(numpy.float64)

# insert target_signal into the dataframe as first column, and initialize it with all 0s
# according to "Data Set Information", the top 36499 rows are signal events, followed by 93565 rows of background events
df_train.insert(0, 'target_signal', 0)
df_train.loc[:36498, 'target_signal'] = 1

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_027_miniboone_particle_identification.csv', index = False)
