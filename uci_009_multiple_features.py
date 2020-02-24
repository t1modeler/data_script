#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Multiple+Features

# if the file is on your local device, change url into local file path, e.g., '‪D:\local_file.data'
url_mfeat_fac = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-fac'
url_mfeat_fou = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-fou'
url_mfeat_kar = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-kar'
url_mfeat_mor = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-mor'
url_mfeat_pix = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-pix'
url_mfeat_zer = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-zer'

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
data_mfeat_fac = download_file(url_mfeat_fac) if url_mfeat_fac.startswith('http') else url_mfeat_fac
data_mfeat_fou = download_file(url_mfeat_fou) if url_mfeat_fou.startswith('http') else url_mfeat_fou
data_mfeat_kar = download_file(url_mfeat_kar) if url_mfeat_kar.startswith('http') else url_mfeat_kar
data_mfeat_mor = download_file(url_mfeat_mor) if url_mfeat_mor.startswith('http') else url_mfeat_mor
data_mfeat_pix = download_file(url_mfeat_pix) if url_mfeat_pix.startswith('http') else url_mfeat_pix
data_mfeat_zer = download_file(url_mfeat_zer) if url_mfeat_zer.startswith('http') else url_mfeat_zer

# generate column names for each of the 6 files
columns_fac = ['profile_correlations_'       + str(i + 1).zfill(3) for i in range(216)]
columns_fou = ['fourier_coefficients_'       + str(i + 1).zfill(3) for i in range( 76)]
columns_kar = ['karhunen_love_coefficients_' + str(i + 1).zfill(3) for i in range( 64)]
columns_mor = ['morphological_'              + str(i + 1).zfill(3) for i in range(  6)]
columns_pix = ['pixel_averages_'             + str(i + 1).zfill(3) for i in range(240)]
columns_zer = ['zernike_moments_'            + str(i + 1).zfill(3) for i in range( 47)]

# convert flat files into pandas dataframes
df_fac = pandas.read_csv(data_mfeat_fac, delimiter = '\s+', header = None, names = columns_fac, index_col = False)
df_fou = pandas.read_csv(data_mfeat_fou, delimiter = '\s+', header = None, names = columns_fou, index_col = False)
df_kar = pandas.read_csv(data_mfeat_kar, delimiter = '\s+', header = None, names = columns_kar, index_col = False)
df_mor = pandas.read_csv(data_mfeat_mor, delimiter = '\s+', header = None, names = columns_mor, index_col = False)
df_pix = pandas.read_csv(data_mfeat_pix, delimiter = '\s+', header = None, names = columns_pix, index_col = False)
df_zer = pandas.read_csv(data_mfeat_zer, delimiter = '\s+', header = None, names = columns_zer, index_col = False)

# generate class variable, according to "Data Set Information"
# the first 200 patterns are of class `0', followed by sets of 200 patterns for each of the classes `1' - `9'
list_target = ['0'] * 200 + ['1'] * 200 + ['2'] * 200 + ['3'] * 200 + ['4'] * 200 + \
              ['5'] * 200 + ['6'] * 200 + ['7'] * 200 + ['8'] * 200 + ['9'] * 200
df_target = pandas.DataFrame(list_target, columns = ['handwritten_numeral'])

# merge target handwritten_numeral with 6 feature sets
df_target = df_target.merge(df_fac, how = 'left', left_index = True, right_index = True)
df_target = df_target.merge(df_fou, how = 'left', left_index = True, right_index = True)
df_target = df_target.merge(df_kar, how = 'left', left_index = True, right_index = True)
df_target = df_target.merge(df_mor, how = 'left', left_index = True, right_index = True)
df_target = df_target.merge(df_pix, how = 'left', left_index = True, right_index = True)
df_target = df_target.merge(df_zer, how = 'left', left_index = True, right_index = True)

# in this example, we merely want to distinguish 5 from other numerals, so we set the target as 1 = '5' and 0 = 'other numerals'
# and we insert target_numeral_5 into the dataframe as the first column
df_target.insert(0, 'target_numeral_5', df_target['handwritten_numeral'].apply(lambda x: 1 if x == '5' else 0))

# drop handwritten_numeral column
df_target = df_target.drop('handwritten_numeral', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_target.to_csv('uci_009_multiple_features.csv', index = False)
