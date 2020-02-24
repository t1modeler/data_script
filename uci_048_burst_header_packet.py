#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Burst+Header+Packet+%28BHP%29+flooding+attack+on+Optical+Burst+Switching+%28OBS%29+Network

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00404/OBS-Network-DataSet_2_Aug27.arff'

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
    'Node',
    'Utilised Bandwith Rate',
    'Packet Drop Rate',
    'Full_Bandwidth',
    'Average_Delay_Time_Per_Sec',
    'Percentage_Of_Lost_Pcaket_Rate',
    'Percentage_Of_Lost_Byte_Rate',
    'Packet Received  Rate',
    'of Used_Bandwidth',
    'Lost_Bandwidth',
    'Packet Size_Byte',
    'Packet_Transmitted',
    'Packet_Received',
    'Packet_lost',
    'Transmitted_Byte',
    'Received_Byte',
    '10-Run-AVG-Drop-Rate',
    '10-Run-AVG-Bandwith-Use',
    '10-Run-Delay',
    'Node Status',
    'Flood Status',
    'Class']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, header = None, names = columns, na_values = '?', quotechar = "'", skiprows = 26)

# drop variables which are not for modeling
df_train = df_train.drop(['Node', 'Node Status'], axis = 1)

# the target variable, inserted into the dataframe as the first column, and drop the original Class variable
# set Class = Block to 1 and Class = other values to 0
df_train.insert(0, 'target_Class', df_train['Class'].apply(lambda x: 1 if x == 'Block' else 0))
df_train = df_train.drop('Class', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_048_burst_header_packet.csv', index = False)
