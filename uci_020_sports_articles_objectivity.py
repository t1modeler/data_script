#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import zipfile
import xml
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Sports+articles+for+objectivity+analysis

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00450/SportsArticles.zip'

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

# unzip the downloaded file, and get data from features.xls
# do not use pandas.read_excel(), because features.xls is actually an xml document
with zipfile.ZipFile(data_train) as myzip:
    with myzip.open('features.xls') as myfile:
        parsedXML = xml.etree.ElementTree.parse(myfile)
        worksheet = parsedXML.findall('{urn:schemas-microsoft-com:office:spreadsheet}Worksheet')[0]
        table = worksheet.find('{urn:schemas-microsoft-com:office:spreadsheet}Table')
        rows = table.findall('{urn:schemas-microsoft-com:office:spreadsheet}Row')

        total_data = []
        column_names = []
        row_number = 1
        for row in rows:
            cells = row.findall('{urn:schemas-microsoft-com:office:spreadsheet}Cell')
            if row_number == 1:
                for cell in cells:
                    column_names.append(cell.find('{urn:schemas-microsoft-com:office:spreadsheet}Data').text)
            else:
                data_records = []
                for cell in cells:
                    node_data = cell.find('{urn:schemas-microsoft-com:office:spreadsheet}Data')
                    if node_data.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Type') == 'Number':
                        node_data_value = float(node_data.text)
                    else:
                        node_data_value = node_data.text
                    data_records.append(node_data_value)

                total_data.append(dict(zip(column_names, data_records)))

            row_number += 1

# convert the total_data list into pandas dataframe
df_train = pandas.DataFrame(total_data)

# drop variables which are not for modedling
df_train = df_train.drop(['TextID', 'URL'], axis = 1)

# the target variable, 1 = objective and 0 = subjective, inserted as first column
df_train.insert(0, 'target_Label', df_train['Label'].apply(lambda x: 1 if x == 'objective' else 0))
df_train = df_train.drop('Label', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_020_sports_articles_objectivity.csv', index = False)
