#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://sci2s.ugr.es/keel/dataset.php?cod=85

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://sci2s.ugr.es/keel/dataset/data/classification/sonar.zip'

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
    'Band1',
    'Band2',
    'Band3',
    'Band4',
    'Band5',
    'Band6',
    'Band7',
    'Band8',
    'Band9',
    'Band10',
    'Band11',
    'Band12',
    'Band13',
    'Band14',
    'Band15',
    'Band16',
    'Band17',
    'Band18',
    'Band19',
    'Band20',
    'Band21',
    'Band22',
    'Band23',
    'Band24',
    'Band25',
    'Band26',
    'Band27',
    'Band28',
    'Band29',
    'Band30',
    'Band31',
    'Band32',
    'Band33',
    'Band34',
    'Band35',
    'Band36',
    'Band37',
    'Band38',
    'Band39',
    'Band40',
    'Band41',
    'Band42',
    'Band43',
    'Band44',
    'Band45',
    'Band46',
    'Band47',
    'Band48',
    'Band49',
    'Band50',
    'Band51',
    'Band52',
    'Band53',
    'Band54',
    'Band55',
    'Band56',
    'Band57',
    'Band58',
    'Band59',
    'Band60',
    'Type']

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('sonar.dat') as myfile:
        df_train = pandas.read_csv(myfile, header = None, names = columns, skiprows = 65, low_memory = False, skipinitialspace = True)

# the target variable, inserted into the dataframe as the first column, and drop the original Type variable
# set Type = M (Mine) to 1 and Type = R (Rock) to 0
df_train.insert(0, 'target_Type', df_train['Type'].apply(lambda x: 1 if x == 'M' else 0))
df_train = df_train.drop('Type', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('keel_002_sonar_mines_vs_rocks.csv', index = False)
