#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import numpy
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

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

# good_or_bad is the original target variable, which will be converted into 0 or 1 later
columns = [
    'Status_of_existing_checking_account',
    'Duration_in_month',
    'Credit_history',
    'Purpose',
    'Credit_amount',
    'Savings_account_bonds',
    'Present_employment_since',
    'Installment_rate_in_percentage_of_disposable_income',
    'Personal_status_and_sex',
    'Other_debtors_guarantors',
    'Present_residence_since',
    'Property',
    'Age_in_years',
    'Other_installment_plans',
    'Housing',
    'Number_of_existing_credits_at_this_bank',
    'Job',
    'Number_of_people_being_liable_to_provide_maintenance_for',
    'Telephone',
    'foreign worker',
    'good_or_bad']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, delimiter = '\s+', header = None, names = columns, index_col = False)

# convert Status_of_existing_checking_account to numeric values, because the underlying categorical values can be binned when modeling
checking_account_dict = {
    'A11': 0,
    'A12': 1,
    'A13': 2,
    'A14': numpy.nan}
df_train['Status_of_existing_checking_account'] = df_train['Status_of_existing_checking_account'].apply(lambda x: checking_account_dict[x])

# convert Savings_account_bonds to numeric values, because the underlying categorical values can be binned when modeling
savings_account_bonds_dict = {
    'A61': 0,
    'A62': 1,
    'A63': 2,
    'A64': 3,
    'A65': numpy.nan}
df_train['Savings_account_bonds'] = df_train['Savings_account_bonds'].apply(lambda x: savings_account_bonds_dict[x])

# convert Present_employment_since to numeric values, because the underlying categorical values can be binned when modeling
present_employment_since_dict = {
    'A71': 0,
    'A72': 1,
    'A73': 2,
    'A74': 3,
    'A75': 4}
df_train['Present_employment_since'] = df_train['Present_employment_since'].apply(lambda x: present_employment_since_dict[x])

# the target variable, we insert target_good_or_bad into the dataframe as the first column (1 = bad, 0 = good)
# and drop the original good_or_bad column
df_train.insert(0, 'target_good_or_bad', df_train['good_or_bad'].apply(lambda x: 1 if x == 2 else 0))
df_train = df_train.drop('good_or_bad', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_013_german_credit_data.csv', index = False)
