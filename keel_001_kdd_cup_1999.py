#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://sci2s.ugr.es/keel/dataset.php?cod=196

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://sci2s.ugr.es/keel/dataset/data/classification/kddcup.zip'

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

columns = [
    'Atr-0',
    'Atr-1',
    'Atr-2',
    'Atr-3',
    'Atr-4',
    'Atr-5',
    'Atr-6',
    'Atr-7',
    'Atr-8',
    'Atr-9',
    'Atr-10',
    'Atr-11',
    'Atr-12',
    'Atr-13',
    'Atr-14',
    'Atr-15',
    'Atr-16',
    'Atr-17',
    'Atr-18',
    'Atr-19',
    'Atr-20',
    'Atr-21',
    'Atr-22',
    'Atr-23',
    'Atr-24',
    'Atr-25',
    'Atr-26',
    'Atr-27',
    'Atr-28',
    'Atr-29',
    'Atr-30',
    'Atr-31',
    'Atr-32',
    'Atr-33',
    'Atr-34',
    'Atr-35',
    'Atr-36',
    'Atr-37',
    'Atr-38',
    'Atr-39',
    'Atr-40',
    'Class']

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('kddcup.dat') as myfile:
        df_train = pandas.read_csv(myfile, header = None, names = columns, skiprows = 46, low_memory = False)

# the target variable, inserted into the dataframe as the first column, and drop the original Class variable
# set Class = normal. to 1 and Class = other values to 0
df_train.insert(0, 'target_Class', df_train['Class'].apply(lambda x: 1 if x == 'normal.' else 0))
df_train = df_train.drop('Class', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('keel_001_kdd_cup_1999.csv', index = False)
