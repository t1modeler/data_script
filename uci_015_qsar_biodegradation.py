#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'

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

# experimental is the original target variable, which will be converted into 0 or 1 later
columns = [
    'SpMax_L',
    'J_Dz(e)',
    'nHM',
    'F01[N-N]',
    'F04[C-N]',
    'NssssC',
    'nCb-',
    'C%',
    'nCp',
    'nO',
    'F03[C-N]',
    'SdssC',
    'HyWi_B(m)',
    'LOC',
    'SM6_L',
    'F03[C-O]',
    'Me',
    'Mi',
    'nN-N',
    'nArNO2',
    'nCRX3',
    'SpPosA_B(p)',
    'nCIR',
    'B01[C-Br]',
    'B03[C-Cl]',
    'N-073',
    'SpMax_A',
    'Psi_i_1d',
    'B04[C-Br]',
    'SdO',
    'TI2_L',
    'nCrt',
    'C-026',
    'F02[C-N]',
    'nHDon',
    'SpMax_B(m)',
    'Psi_i_A',
    'nN',
    'SM6_B(m)',
    'nArCOOR',
    'nX',
    'experimental']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, delimiter = ';', header = None, names = columns, index_col = False)

# the target variable, 1 = RB and 0 = NRB
# we insert target_experimental into the dataframe as the first column, and drop the original experimental column
df_train.insert(0, 'target_experimental', df_train['experimental'].apply(lambda x: 1 if x == 'RB' else 0))
df_train = df_train.drop('experimental', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_015_qsar_biodegradation.csv', index = False)
