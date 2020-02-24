#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Cylinder+Bands

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cylinder-bands/bands.data'

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

# a line of data was accidentally separated into 2 lines, here we correct it by replacing ',\n' with ','
content = io.BytesIO(data_train.read().decode().replace(',\n', ',').encode())

# band_type is the original target variable, which will be converted into 0 or 1 later
columns = [
    'timestamp',
    'cylinder_number',
    'customer',
    'job_number',
    'grain_screened',
    'ink_color',
    'proof_on_ctd_ink',
    'blade_mfg',
    'cylinder_division',
    'paper_type',
    'ink_type',
    'direct_steam',
    'solvent_type',
    'type_on_cylinder',
    'press_type',
    'press',
    'unit_number',
    'cylinder_size',
    'paper_mill_location',
    'plating_tank',
    'proof_cut',
    'viscosity',
    'caliper',
    'ink_temperature',
    'humifity',
    'roughness',
    'blade_pressure',
    'varnish_pct',
    'press_speed',
    'ink_pct',
    'solvent_pct',
    'ESA_Voltage',
    'ESA_Amperage',
    'wax',
    'hardener',
    'roller_durometer',
    'current_density',
    'anode_space_ratio',
    'chrome_content',
    'band_type']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(content, header = None, names = columns, index_col = False, na_values = '?')

# drop those variables which are not for modeling
df_train = df_train.drop(['timestamp', 'cylinder_number', 'customer', 'ink_color', 'cylinder_division', 'direct_steam'], axis = 1)

# these variables contain case-insensitive words, we convert them to upper letters
df_train['paper_type'] = df_train['paper_type'].str.upper()
df_train['ink_type'] = df_train['ink_type'].str.upper()
df_train['type_on_cylinder'] = df_train['type_on_cylinder'].str.upper()
df_train['cylinder_size'] = df_train['cylinder_size'].str.upper()
df_train['paper_mill_location'] = df_train['paper_mill_location'].str.upper()

# finally the target variable, 0 = noband and 1 = band
df_train['target_band_type'] = df_train['band_type'].apply(lambda x: 0 if x == 'noband' else 1)
df_train = df_train.drop('band_type', axis = 1)

# re-order the columns so that target_band_type becomes the first column
df_train = df_train[[
    'target_band_type',
    'job_number',
    'grain_screened',
    'proof_on_ctd_ink',
    'blade_mfg',
    'paper_type',
    'ink_type',
    'solvent_type',
    'type_on_cylinder',
    'press_type',
    'press',
    'unit_number',
    'cylinder_size',
    'paper_mill_location',
    'plating_tank',
    'proof_cut',
    'viscosity',
    'caliper',
    'ink_temperature',
    'humifity',
    'roughness',
    'blade_pressure',
    'varnish_pct',
    'press_speed',
    'ink_pct',
    'solvent_pct',
    'ESA_Voltage',
    'ESA_Amperage',
    'wax',
    'hardener',
    'roller_durometer',
    'current_density',
    'anode_space_ratio',
    'chrome_content']]

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_005_cylinder_bands.csv', index = False)
