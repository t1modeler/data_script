#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/HCC+Survival

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00423/hcc-survival.zip'

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

columns = [
    'Gender',
    'Symptoms',
    'Alcohol',
    'Hepatitis B Surface Antigen',
    'Hepatitis B e Antigen',
    'Hepatitis B Core Antibody',
    'Hepatitis C Virus Antibody',
    'Cirrhosis',
    'Endemic Countries',
    'Smoking',
    'Diabetes',
    'Obesity',
    'Hemochromatosis',
    'Arterial Hypertension',
    'Chronic Renal Insufficiency',
    'Human Immunodeficiency Virus',
    'Nonalcoholic Steatohepatitis',
    'Esophageal Varices',
    'Splenomegaly',
    'Portal Hypertension',
    'Portal Vein Thrombosis',
    'Liver Metastasis',
    'Radiological Hallmark',
    'Age at diagnosis',
    'Grams of Alcohol per day',
    'Packs of cigarets per year',
    'Performance Status',
    'Encefalopathy degree',
    'Ascites degree',
    'International Normalised Ratio',
    'Alpha-Fetoprotein (ng/mL)',
    'Haemoglobin (g/dL)',
    'Mean Corpuscular Volume (fl)',
    'Leukocytes(G/L)',
    'Platelets (G/L)',
    'Albumin (mg/dL)',
    'Total Bilirubin(mg/dL)',
    'Alanine transaminase (U/L)',
    'Aspartate transaminase (U/L)',
    'Gamma glutamyl transferase (U/L)',
    'Alkaline phosphatase (U/L)',
    'Total Proteins (g/dL)',
    'Creatinine (mg/dL)',
    'Number of Nodules',
    'Major dimension of nodule (cm)',
    'Direct Bilirubin (mg/dL)',
    'Iron (mcg/dL)',
    'Oxygen Saturation (%)',
    'Ferritin (ng/mL)',
    'Class']

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('hcc-survival/hcc-data.txt') as myfile:
        df_train = pandas.read_csv(myfile, header = None, names = columns, na_values = '?')

# the target variable, inserted into the dataframe as the first column, and drop the original Class variable
df_train.insert(0, 'target_Class', df_train['Class'])
df_train = df_train.drop('Class', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_051_hcc_survival.csv', index = False)
