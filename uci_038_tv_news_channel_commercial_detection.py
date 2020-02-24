#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import sklearn.datasets
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/TV+News+Channel+Commercial+Detection+Dataset

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00326/TV_News_Channel_Commercial_Detection_Dataset.zip'

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

audio_word_columns = ['word_' + str(i + 1).zfill(4) for i in range(4000)]

columns = [
    'Shot_Length',
    'Motion_Distribution_Mean',
    'Motion_Distribution_Variance',
    'Frame_Difference_Distribution_Mean',
    'Frame_Difference_Distribution_Variance',
    'Short_time_energy_Mean',
    'Short_time_energy_Variance',
    'ZCR_Mean',
    'ZCR_Variance',
    'Spectral_Centroid_Mean',
    'Spectral_Centroid_Variance',
    'Spectral_Roll_off_Mean',
    'Spectral_Roll_off_Variance',
    'Spectral_Flux_Mean',
    'Spectral_Flux_Variance',
    'Fundamental_Frequency_Mean',
    'Fundamental_Frequency_Variance',
    'Motion_Distribution_01',
    'Motion_Distribution_02',
    'Motion_Distribution_03',
    'Motion_Distribution_04',
    'Motion_Distribution_05',
    'Motion_Distribution_06',
    'Motion_Distribution_07',
    'Motion_Distribution_08',
    'Motion_Distribution_09',
    'Motion_Distribution_10',
    'Motion_Distribution_11',
    'Motion_Distribution_12',
    'Motion_Distribution_13',
    'Motion_Distribution_14',
    'Motion_Distribution_15',
    'Motion_Distribution_16',
    'Motion_Distribution_17',
    'Motion_Distribution_18',
    'Motion_Distribution_19',
    'Motion_Distribution_20',
    'Motion_Distribution_21',
    'Motion_Distribution_22',
    'Motion_Distribution_23',
    'Motion_Distribution_24',
    'Motion_Distribution_25',
    'Motion_Distribution_26',
    'Motion_Distribution_27',
    'Motion_Distribution_28',
    'Motion_Distribution_29',
    'Motion_Distribution_30',
    'Motion_Distribution_31',
    'Motion_Distribution_32',
    'Motion_Distribution_33',
    'Motion_Distribution_34',
    'Motion_Distribution_35',
    'Motion_Distribution_36',
    'Motion_Distribution_37',
    'Motion_Distribution_38',
    'Motion_Distribution_39',
    'Motion_Distribution_40',
    'Motion_Distribution_41',
    'Frame_Difference_Distribution_01',
    'Frame_Difference_Distribution_02',
    'Frame_Difference_Distribution_03',
    'Frame_Difference_Distribution_04',
    'Frame_Difference_Distribution_05',
    'Frame_Difference_Distribution_06',
    'Frame_Difference_Distribution_07',
    'Frame_Difference_Distribution_08',
    'Frame_Difference_Distribution_09',
    'Frame_Difference_Distribution_10',
    'Frame_Difference_Distribution_11',
    'Frame_Difference_Distribution_12',
    'Frame_Difference_Distribution_13',
    'Frame_Difference_Distribution_14',
    'Frame_Difference_Distribution_15',
    'Frame_Difference_Distribution_16',
    'Frame_Difference_Distribution_17',
    'Frame_Difference_Distribution_18',
    'Frame_Difference_Distribution_19',
    'Frame_Difference_Distribution_20',
    'Frame_Difference_Distribution_21',
    'Frame_Difference_Distribution_22',
    'Frame_Difference_Distribution_23',
    'Frame_Difference_Distribution_24',
    'Frame_Difference_Distribution_25',
    'Frame_Difference_Distribution_26',
    'Frame_Difference_Distribution_27',
    'Frame_Difference_Distribution_28',
    'Frame_Difference_Distribution_29',
    'Frame_Difference_Distribution_30',
    'Frame_Difference_Distribution_31',
    'Frame_Difference_Distribution_32',
    'Frame_Difference_Distribution_33',
    'Text_area_distribution_Mean_01',
    'Text_area_distribution_Mean_02',
    'Text_area_distribution_Mean_03',
    'Text_area_distribution_Mean_04',
    'Text_area_distribution_Mean_05',
    'Text_area_distribution_Mean_06',
    'Text_area_distribution_Mean_07',
    'Text_area_distribution_Mean_08',
    'Text_area_distribution_Mean_09',
    'Text_area_distribution_Mean_10',
    'Text_area_distribution_Mean_11',
    'Text_area_distribution_Mean_12',
    'Text_area_distribution_Mean_13',
    'Text_area_distribution_Mean_14',
    'Text_area_distribution_Mean_15',
    'Text_area_distribution_Mean_16',
    'Text_area_distribution_variance_01',
    'Text_area_distribution_variance_02',
    'Text_area_distribution_variance_03',
    'Text_area_distribution_variance_04',
    'Text_area_distribution_variance_05',
    'Text_area_distribution_variance_06',
    'Text_area_distribution_variance_07',
    'Text_area_distribution_variance_08',
    'Text_area_distribution_variance_09',
    'Text_area_distribution_variance_10',
    'Text_area_distribution_variance_11',
    'Text_area_distribution_variance_12',
    'Text_area_distribution_variance_13',
    'Text_area_distribution_variance_14',
    'Text_area_distribution_variance_15',
    'Text_area_distribution_variance_16'] + audio_word_columns + ['Edge_change_Ratio_Mean', 'Edge_change_Ratio_Variance']

# unzip the downloaded file, and get data files
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('CNN.txt') as myfile:
        df_x, df_y = sklearn.datasets.load_svmlight_file(myfile)

# convert libsvm format files into pandas dataframes
df_x = pandas.DataFrame(df_x.todense(), columns = columns)
df_y = pandas.DataFrame(df_y, columns = ['label'])

# drop the 4000 audio word columns, because there are too many of them
#df_x = df_x.drop(audio_word_columns, axis = 1)

# merge y with x variables
df_total = df_y.merge(df_x, how = 'left', left_index = True, right_index = True)

# convert label values (originally 1 and -1) into 0 and 1
df_total['label'] = df_total['label'].apply(lambda x: 1 if x == 1 else 0)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_total.to_csv('uci_038_tv_news_channel_commercial_detection.csv', index = False)
