#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import numpy
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip'

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
    with myzip.open('dataset_diabetes/diabetic_data.csv') as myfile:
        df_training = pandas.read_csv(myfile, header = 0, index_col = False)

# drop variables which are inappropriate for modeling
df_training = df_training.drop(['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id',
    'admission_source_id', 'diag_1', 'diag_2', 'diag_3'], axis = 1)

# convert age into numeric values because it could be binned when modeling
age_dict = {
    '[0-10)'  :  1,
    '[10-20)' :  2,
    '[20-30)' :  3,
    '[30-40)' :  4,
    '[40-50)' :  5,
    '[50-60)' :  6,
    '[60-70)' :  7,
    '[70-80)' :  8,
    '[80-90)' :  9,
    '[90-100)': 10
    }
df_training['age'] = df_training['age'].apply(lambda x: age_dict[x]).astype(numpy.int64)

# the target variable, inserted into the dataframe as the first column, and drop the original readmitted variable
df_training.insert(0, 'target_readmitted', df_training['readmitted'].apply(lambda x: 0 if x == 'NO' else 1))
df_training = df_training.drop('readmitted', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_training.to_csv('uci_034_diabetes_130_us_hospitals.csv', index = False)
