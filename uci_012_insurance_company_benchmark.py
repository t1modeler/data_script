#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import io
import pandas # install pandas by "pip install pandas", or install Anaconda distribution (https://www.anaconda.com/)

# Warning: the data processing techniques shown below are just for concept explanation, which are not best-proctices

# data set repository
# https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+%28COIL+2000%29

# if the file is on your local device, change url_data_train into local file path, e.g., '‪D:\local_file.data'
url_data_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt'

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

# Number_of_mobile_home_policies is the original target variable, which will be converted into 0 or 1 later
columns = [
    'Customer_Subtype',
    'Number_of_houses',
    'Avg_size_household',
    'Avg_age',
    'Customer_main_type',
    'Roman_catholic',
    'Protestant',
    'Other_religion',
    'No_religion',
    'Married',
    'Living_together',
    'Other_relation',
    'Singles',
    'Household_without_children',
    'Household_with_children',
    'High_level_education',
    'Medium_level_education',
    'Lower_level_education',
    'High_status',
    'Entrepreneur',
    'Farmer',
    'Middle_management',
    'Skilled_labourers',
    'Unskilled_labourers',
    'Social_class_A',
    'Social_class_B1',
    'Social_class_B2',
    'Social_class_C',
    'Social_class_D',
    'Rented_house',
    'Home_owners',
    '1_car',
    '2_cars',
    'No_car',
    'National_Health_Service',
    'Private_health_insurance',
    'Income_<_30',
    'Income_30-45.000',
    'Income_45-75.000',
    'Income_75-122.000',
    'Income_>123.000',
    'Average_income',
    'Purchasing_power_class',
    'Contribution_private_third_party_insurance',
    'Contribution_third_party_insurance_(firms)',
    'Contribution_third_party_insurane_(agriculture)',
    'Contribution_car_policies',
    'Contribution_delivery_van_policies',
    'Contribution_motorcycle/scooter_policies',
    'Contribution_lorry_policies',
    'Contribution_trailer_policies',
    'Contribution_tractor_policies',
    'Contribution_agricultural_machines_policies',
    'Contribution_moped_policies',
    'Contribution_life_insurances',
    'Contribution_private_accident_insurance_policies',
    'Contribution_family_accidents_insurance_policies',
    'Contribution_disability_insurance_policies',
    'Contribution_fire_policies',
    'Contribution_surfboard_policies',
    'Contribution_boat_policies',
    'Contribution_bicycle_policies',
    'Contribution_property_insurance_policies',
    'Contribution_social_security_insurance_policies',
    'Number_of_private_third_party_insurance',
    'Number_of_third_party_insurance_(firms)',
    'Number_of_third_party_insurane_(agriculture)',
    'Number_of_car_policies',
    'Number_of_delivery_van_policies',
    'Number_of_motorcycle/scooter_policies',
    'Number_of_lorry_policies',
    'Number_of_trailer_policies',
    'Number_of_tractor_policies',
    'Number_of_agricultural_machines_policies',
    'Number_of_moped_policies',
    'Number_of_life_insurances',
    'Number_of_private_accident_insurance_policies',
    'Number_of_family_accidents_insurance_policies',
    'Number_of_disability_insurance_policies',
    'Number_of_fire_policies',
    'Number_of_surfboard_policies',
    'Number_of_boat_policies',
    'Number_of_bicycle_policies',
    'Number_of_property_insurance_policies',
    'Number_of_social_security_insurance_policies',
    'Number_of_mobile_home_policies']

# convert flat file into pandas dataframe
df_train = pandas.read_csv(data_train, delimiter = '\s+', header = None, names = columns, index_col = False)

# map Customer_Subtype to it's string values
# because Customer_Subtype is not meant to be numeric values and shouldn't be binned when modeling
customer_subtype_dict = {
     1: 'High Income, expensive child',
     2: 'Very Important Provincials',
     3: 'High status seniors',
     4: 'Affluent senior apartments',
     5: 'Mixed seniors',
     6: 'Career and childcare',
     7: 'Dinkis (double income no kids)',
     8: 'Middle class families',
     9: 'Modern, complete families',
    10: 'Stable family',
    11: 'Family starters',
    12: 'Affluent young families',
    13: 'Young all american family',
    14: 'Junior cosmopolitan',
    15: 'Senior cosmopolitans',
    16: 'Students in apartments',
    17: 'Fresh masters in the city',
    18: 'Single youth',
    19: 'Suburban youth',
    20: 'Etnically diverse',
    21: 'Young urban have-nots',
    22: 'Mixed apartment dwellers',
    23: 'Young and rising',
    24: 'Young, low educated',
    25: 'Young seniors in the city',
    26: 'Own home elderly',
    27: 'Seniors in apartments',
    28: 'Residential elderly',
    29: 'Porchless seniors: no front yard',
    30: 'Religious elderly singles',
    31: 'Low income catholics',
    32: 'Mixed seniors',
    33: 'Lower class large families',
    34: 'Large family, employed child',
    35: 'Village families',
    36: 'Couples with teens Married with children',
    37: 'Mixed small town dwellers',
    38: 'Traditional families',
    39: 'Large religous families',
    40: 'Large family farms',
    41: 'Mixed rurals'}
df_train['Customer_Subtype'] = df_train['Customer_Subtype'].apply(lambda x: customer_subtype_dict[x])

# map Customer_main_type to it's string values
# because Customer_main_type is not meant to be numeric values and shouldn't be binned when modeling
customer_main_type_dict = {
     1: 'Successful hedonists',
     2: 'Driven Growers',
     3: 'Average Family',
     4: 'Career Loners',
     5: 'Living well',
     6: 'Cruising Seniors',
     7: 'Retired and Religeous',
     8: 'Family with grown ups',
     9: 'Conservative families',
    10: 'Farmers'}
df_train['Customer_main_type'] = df_train['Customer_main_type'].apply(lambda x: customer_subtype_dict[x])

# the target variable, we insert target_Number_of_mobile_home_policies into the dataframe as the first column
# and drop the original Number_of_mobile_home_policies column
df_train.insert(0, 'target_Number_of_mobile_home_policies', df_train['Number_of_mobile_home_policies'])
df_train = df_train.drop('Number_of_mobile_home_policies', axis = 1)

# save the dataframe as CSV file, you can zip it, upload it to t1modeler.com, and build a model
df_train.to_csv('uci_012_insurance_company_benchmark.csv', index = False)
