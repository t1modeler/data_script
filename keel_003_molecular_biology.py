#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://sci2s.ugr.es/keel/dataset.php?cod=181

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://sci2s.ugr.es/keel/dataset/data/classification/splice.zip'

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
    'POS1',
    'POS2',
    'POS3',
    'POS4',
    'POS5',
    'POS6',
    'POS7',
    'POS8',
    'POS9',
    'POS10',
    'POS11',
    'POS12',
    'POS13',
    'POS14',
    'POS15',
    'POS16',
    'POS17',
    'POS18',
    'POS19',
    'POS20',
    'POS21',
    'POS22',
    'POS23',
    'POS24',
    'POS25',
    'POS26',
    'POS27',
    'POS28',
    'POS29',
    'POS30',
    'POS31',
    'POS32',
    'POS33',
    'POS34',
    'POS35',
    'POS36',
    'POS37',
    'POS38',
    'POS39',
    'POS40',
    'POS41',
    'POS42',
    'POS43',
    'POS44',
    'POS45',
    'POS46',
    'POS47',
    'POS48',
    'POS49',
    'POS50',
    'POS51',
    'POS52',
    'POS53',
    'POS54',
    'POS55',
    'POS56',
    'POS57',
    'POS58',
    'POS59',
    'POS60',
    'Class']

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('splice.dat') as myfile:
        df_train = pandas.read_csv(myfile, header = None, names = columns, skiprows = 65, low_memory = False, skipinitialspace = True)

# the target variable, inserted into the dataframe as the first column, and drop the original Class variable
# set Class = N to 1 and Class = other values to 0
df_train.insert(0, 'target_Class', df_train['Class'].apply(lambda x: 1 if x == 'N' else 0))
df_train = df_train.drop('Class', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('keel_003_molecular_biology.csv', index = False)
