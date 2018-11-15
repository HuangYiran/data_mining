#-*- coding: utf-8 -*-

# load package
# string 
import re

# math
import pandas as pd
import numpy as np
import scipy as sp
import random

# sys
import sys
import os
import time
import warnings

# date
from datetime import datetime, timedelta

# machine learning
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, gaussian_process, discriminant_analysis
from xgboost import XGBClassifier

# model utils
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection 
from sklearn import model_selection
from sklearn import metrics

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix #??
# = show plots in Jupyter Notebook browser
#%matplotlib inline 
mpl.style.use('ggplot') #??
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8 #??

# show all columns
from IPython.display import display
pd.options.display.max_columns = None

# memory manage
import gc

# logging
import logging 

# other
import tqdm as tqdm

# self define
sys.path.append('../../utils/')

def main():
    # set logging
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler('../log/extract_features.log')
    handler.setLevel(logging.INFO)
    formater = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formater)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    #logger.info('This is a log info')
    # read files
    logger.info('Read Data')
    lb_wsp_2014 = pd.read_csv('../data/Autohaus_weeber/leonberg_werkstattposten_2014.csv', sep = ';')
    lb_wsp_2015 = pd.read_csv('../data/Autohaus_weeber/leonberg_werkstattposten_2015.csv', sep = ';')
    lb_wsp_2016 = pd.read_csv('../data/Autohaus_weeber/leonberg_werkstattposten_2016.csv', sep = ';')
    lb_wsp_2017 = pd.read_csv('../data/Autohaus_weeber/leonberg_werkstattposten_2017.csv', sep = ';')
    lb_wsp_2018 = pd.read_csv('../data/Autohaus_weeber/leonberg_werkstattposten_2018.csv', sep = ';')
    std_wsp_2014 = pd.read_csv('../data/Autohaus_weeber/weil_der_stadt_werkstattposten_2014.csv', sep = ';')
    std_wsp_2015 = pd.read_csv('../data/Autohaus_weeber/weil_der_stadt_werkstattposten_2015.csv', sep = ';')
    std_wsp_2016 = pd.read_csv('../data/Autohaus_weeber/weil_der_stadt_werkstattposten_2016.csv', sep = ';')
    std_wsp_2017 = pd.read_csv('../data/Autohaus_weeber/weil_der_stadt_werkstattposten_2017.csv', sep = ';')
    std_wsp_2018 = pd.read_csv('../data/Autohaus_weeber/weil_der_stadt_werkstattposten_2018.csv', sep = ';')
    # cat
    d1 = lb_wsp_2014.copy()
    d2 = lb_wsp_2015.copy()
    d3 = lb_wsp_2016.copy()
    d4 = lb_wsp_2017.copy()
    d5 = lb_wsp_2018.copy()

    d6 = std_wsp_2014.copy()
    d7 = std_wsp_2015.copy()
    d8 = std_wsp_2016.copy()
    d9 = std_wsp_2017.copy()
    d10 = std_wsp_2018.copy()
    # for training we use data from 14 - 17 in lb_wsp
    train1 = pd.concat([d1,d2,d3,d4,d5], 0)
    train2 = pd.concat([d6,d7,d8,d9,d10], 0)
    train1['Autohaus'] = 'leonberg'
    train2['Autohaus'] = 'weil'
    # because there exists reused Auftragsnummer in different Autohaus station. So we add some sign to the Auftragsnummer
    # in each Autohaus station
    train1['Auftragsnummer'] = 'A' + train1['Auftragsnummer']
    train2['Auftragsnummer'] = 'B' + train2['Auftragsnummer']
    train = pd.concat([train1, train2], 0)
    # remove the outlier in the Teile-Nr
    logger.info('Removing the outlier in the Teile-Nr')
    # remove the na wert
    df = train[train['Teile-Nr'].isna().map(lambda x: not x)]
    # remove the value short value
    df = df[df['Teile-Nr'].map(lambda x: False if len(x) < 4 else True)]
    # remove the value that doesn't contain number: 30593 of 593527 in train1(5%)
    df = df[df['Teile-Nr'].map(lambda x: True if re.search('\d', x) else False)]
    # remove the value that doesn't contrain adjoining number(min 2 number): 594 of 562934 in train1(0.1%)
    df = df[df['Teile-Nr'].map(lambda x: True if re.search('\d\d', x) else False)]
    # use findall instead of search, because, the df here should contain adjoining number, otherwise it's wrong
    df['Gruppe-Nr'] = df['Teile-Nr'].map(lambda x: re.findall('\d\d', x)[0][0])
    # clean the memory
    del train1, train2, lb_wsp_2014, lb_wsp_2015, lb_wsp_2016, lb_wsp_2017, lb_wsp_2018
    del std_wsp_2014, std_wsp_2015, std_wsp_2016, std_wsp_2017, std_wsp_2018
    del d1, d2, d3, d4, d5, d6, d7, d8, d9, d10
    gc.collect()
    cf = df.copy()
    # drop duplicate
    cf = cf.drop_duplicates()
    # find out the confused data. For the same Auftragsnummer, exists more than one value in the other attribute.
    # here is AWSAU310019, BWSAU386471, BWSAU435051
    logger.info('remove confused data in Auftragsnummer')
    cf = cf.drop(cf[cf['Auftragsnummer'] == 'AWSAU310019'].index, axis= 0) # 8 items
    cf = cf.drop(cf[cf['Auftragsnummer'] == 'BWSAU386471'].index, axis= 0) # 3 items
    cf = cf.drop(cf[cf['Auftragsnummer'] == 'BWSAU435051'].index, axis= 0) # 2 items
    cf = cf.drop(cf[cf['Auftragsnummer'] == 'BWSAU271939'].index, axis= 0) # ? items
    # remove wierd auftragsnummer
    cf = cf.drop(cf[cf['Auftragsnummer'] == '103K'].index) #
    cf = cf.drop(cf[cf['Auftragsnummer'] == '77KW'].index) #
    # remove na value in Fahrgestellnummer
    logger.info('removing confused data in Fahrgestellnummer')
    cf = cf.drop(cf[cf['Fahrgestellnummer'].isna()].index, axis = 0) #
    # remove the Fahrgestellnumer that only contain number, not tested!!!!
    cf = cf.drop(cf[cf['Fahrgestellnummer'].map(lambda x: False if re.search('[a-zA-Z]', x) else True)].index, axis = 0)
    # remove the items, which day of the Auftragsdatum lareger than 31, not tested!!!!
    logger.info('removing confused data in Auftragsdatum')
    cf = cf.drop(cf[cf['Auftragsdatum'].map(lambda x: int(x[0:2])) > 31].index, axis = 0)
    # remove the items, which length of the Auftragsdatum shorter than 10, not tested!!!!
    cf = cf.drop(cf[cf['Auftragsdatum'].map(lambda x:True if len(x) < 10 else False)].index, axis = 0)
    # remove the items, which year of the Auftragsdatum smaller than 2013, not tested!!!!
    cf = cf.drop(cf[cf['Auftragsdatum'].map(lambda x: int(x[6:]) < 2013)].index, axis = 0)
    df = cf
    del cf
    gc.collect()
    logger.info('creating feature: total-nr, num_teile, num_act')
    gn = df.copy()
    gn = toAuftragTable(gn, 'Gruppe-Nr', 'Auftragsnummer') # number: 245018
    aw = train.copy() # can't use df here, because df delete all the na value in Teile-Nr
    # remove the items, that contain ziffer in the AW-Nr, set data type to int
    #logger.info('removing outlier in AW-Nr')
    aw['AW-Nr'][aw['AW-Nr'].isnull()] = 0 # set null value to 0
    aw['AW-Nr'][aw['AW-Nr'] == '99999999'] = 0 # set 99999999 to 0
    aw['AW-Nr'] = aw['AW-Nr'].map(lambda x: re.sub(' ', '', str(x)))# remove space between the number
    del_index = aw[aw['AW-Nr'].map(lambda x: bool(re.search('[a-zA-Z]', str(x))))].index # remove fake data
    aw = aw.drop(del_index) 
    aw['AW-Nr'] = aw['AW-Nr'].astype('float')
    # to auftrag table, get the feature: total AW-Nr for each auftrag
    tmp = aw[['AW-Nr', 'Auftragsnummer']]
    ag = tmp.groupby('Auftragsnummer', as_index = False)
    out = []
    for name, gruppe in ag:
        out.append(pd.DataFrame({'Auftragsnummer': [gruppe['Auftragsnummer'].iloc[0]], 'total AW': [gruppe['AW-Nr'].sum()]}))
    aw = pd.concat(out).reset_index()[['Auftragsnummer', 'total AW']] # 缺分段
    # drop the wierd Auftragsnummer
    aw = aw.drop(aw[aw['Auftragsnummer'] == '103K'].index)
    aw = aw.drop(aw[aw['Auftragsnummer'] == '77KW'].index)
    # the lengths of aw and gn are different, because in some auftrag, the customer do not buy anything. 
    # These part of auftrags are not included in gn, but gn is not a subset of aw. 
    # because some AW-Nr in aw are wrong, we have deleted these items in aw. 
    # Therefore, if we want to merge aw and gn, we have to use the method inner
    agn = pd.merge(aw, gn, how = 'inner', on = 'Auftragsnummer')
    aw = train.copy()
    # remove the items, that contain ziffer in the AW-Nr, set data type to int
    aw['AW-Nr'][aw['AW-Nr'].isnull()] = 0 # set null value to 0
    aw['AW-Nr'][aw['AW-Nr'] == '99999999'] = 0 # set 99999999 to 0
    aw['AW-Nr'] = aw['AW-Nr'].map(lambda x: re.sub(' ', '', str(x)))# remove space between the number
    del_index = aw[aw['AW-Nr'].map(lambda x: bool(re.search('[a-zA-Z]', str(x))))].index # remove fake data
    aw = aw.drop(del_index) 
    aw['AW-Nr'] = aw['AW-Nr'].astype('float')
    # to auftrag table, get the feature: total AW-Nr for each auftrag
    tmp = aw[['AW-Nr', 'Auftragsnummer']]
    ag = tmp.groupby('Auftragsnummer', as_index = False)
    out = []
    for name, gruppe in ag:
        out.append(pd.DataFrame({'Auftragsnummer': [gruppe['Auftragsnummer'].iloc[0]], 'num_act': [len(gruppe)]}))
    aw = pd.concat(out).reset_index()[['Auftragsnummer', 'num_act']] # 缺分段
    # drop the wierd Auftragsnummer
    aw = aw.drop(aw[aw['Auftragsnummer'] == '103K'].index)
    aw = aw.drop(aw[aw['Auftragsnummer'] == '77KW'].index)
    #agn = pd.merge(aw, agn, how = 'inner', on = 'Auftragsnummer')
    agn = pd.merge(aw, agn, how = 'inner', on = 'Auftragsnummer')
    agn['num_teile'] = agn['Gruppe-Nr'].map(lambda x: len(x.split(';')))
    agn[['total AW', 'num_act']].corr() 
    del aw, gn
    gc.collect()
    tmp = df[['Auftragsnummer', 'KM-Stand','Markencode', 'Lagerortcode', 'Auftragsdatum', 'Adressanredecode', 'Motorcode', 'Fahrzeugmodellnummer', 'Modell', 'Typ', 
              'Getriebecode', 'Gewicht', 'Leistung (KW)', 'Fahrgestellnummer', 'Erstzulassungsdatum']]
    tmp = tmp.drop_duplicates()
    tmp = pd.merge(agn, tmp, how = 'inner', on = 'Auftragsnummer') # 244608
    df = tmp
    del tmp
    gc.collect()
    logger.info('creating features: year, month, day')
    md = df.copy()
    md['year'] = md['Auftragsdatum'].map(lambda x: x[6:])
    md['month'] = md['Auftragsdatum'].map(lambda x: x[3:5])
    md['day'] = md['Auftragsdatum'].map(lambda x: x[0:2])
    df = md
    del md
    gc.collect()
    #tmp = train[['Auftragsdatum', 'Erstzulassungsdatum']]
    logger.info('create features: age_auto, age_autohaus')
    tmp = df.copy()
    tmp['Auftragsdatum'] = pd.to_datetime(tmp['Auftragsdatum'], dayfirst=True)
    tmp['Erstzulassungsdatum'] = pd.to_datetime(tmp['Erstzulassungsdatum'], dayfirst=True)
    tmp['age_auto'] = tmp['Auftragsdatum'] - tmp['Erstzulassungsdatum']
    tmp['age_autohaus'] = tmp['Auftragsdatum'].map(lambda x: x.year) - 2013
    df = tmp
    del tmp
    gc.collect()
    df['KM-Stand'] = df['KM-Stand'].map(lambda x: float(re.sub(',', '.', x)))
    td = df.copy()
    td = td[['Fahrgestellnummer', 'Auftragsdatum', 'KM-Stand','num_act', 'total AW', 'num_teile', 'year']]
    #1C8FYN8O63T585854 8
    # code to extract new features, remmenber to remove counter
    out = []
    counter = 0
    logger.info('create features: time relative:')
    total_group = len(td['Fahrgestellnummer'].unique())
    for name, group in td.groupby('Fahrgestellnummer', as_index = False):
        counter = counter + 1
        if counter % 1000 == 0:
            num_group = counter + 1
            logger.info('number of group: {} [{}/{} ({:0f}%)]'.format(num_group, num_group, total_group, 100*0.1*num_group/total_group))
            tmp_df = pd.concat(out)
            tmp_df.to_pickle('tmp.pkl')
            #break
        #if name == '1C8FYN8O63T585854':
        #    tmp = group.reindex()
        #    print(tmp.columns)
        #    print(tmp.index)
        #    # time distance since/to last/next come
        # create feature: time distance from last coming
        tmp = group
        if len(tmp) > 1:
            tmp['times'] = len(tmp)
            tmp = tmp.sort_values(by = 'Auftragsdatum').reset_index().iloc[:,1:] # not tested
            # create feature: time distance from last coming
            time_shift_forward = pd.concat([pd.Series(datetime(2010, 1, 1)), tmp['Auftragsdatum'][:-1]])
            time_shift_forward.index = tmp.index
            tmp['time_distance_from_last'] = tmp['Auftragsdatum'] - time_shift_forward
            tmp['time_distance_from_last'] = tmp['time_distance_from_last'].clip(lower = timedelta(0), upper = timedelta(365)) # clip
            # create feature: mean time distance in group, should i clip the data first????
            tmp['mean_time_distance_in_group'] = tmp['time_distance_from_last'].mean()
            # create feature: max time distance in group
            tmp['max_time_distance_in_group'] = tmp['time_distance_from_last'].max()
            # create feature: min time distance in group
            tmp['min_time_distance_in_group'] = tmp['time_distance_from_last'].min()
            # create feature: time distance to next coming
            time_distance_to_next = pd.concat([tmp['time_distance_from_last'][1:], pd.Series(np.NaN)])
            time_distance_to_next.index = tmp.index
            tmp['time_distance_to_next'] = time_distance_to_next
            # reset the first value and last value
            tmp.iloc[0, tmp.columns.get_loc('time_distance_from_last')] = np.NaN
            tmp.iloc[-1, tmp.columns.get_loc('time_distance_to_next')] = np.NaN
            # create feature: km stand distance from last coming
            km_shift_forward = pd.concat([pd.Series(0.0), tmp['KM-Stand'][:-1]])
            km_shift_forward.index = tmp.index
            tmp['km_distance_from_last'] = tmp['KM-Stand'] -km_shift_forward
            tmp['km_distance_from_last'] = tmp['km_distance_from_last'].abs() # correct the negative value in a stupid way!!!
            tmp['km_distance_from_last'][tmp['km_distance_from_last'] > 10000] = 10000 # clip
            # create feature: mean km stand distance, not test
            tmp['mean_km_distance_in_group'] = tmp['km_distance_from_last'].mean() 
            # create feature: max km stand distance, not test
            tmp['max_km_distance_in_group'] = tmp['km_distance_from_last'].max()
            # create feature: min km stand distance, not test
            tmp['min_km_distance_in_group'] = tmp['km_distance_from_last'].min()
            # create feature: km stand distance to next coming
            km_distance_to_next = pd.concat([tmp['km_distance_from_last'][1:], pd.Series(0.0)])
            km_distance_to_next.index = tmp.index
            tmp['km_distance_to_next'] = km_distance_to_next
            # reset the first value and last value
            tmp.iloc[0, tmp.columns.get_loc('km_distance_from_last')] = np.NaN
            tmp.iloc[-1, tmp.columns.get_loc('km_distance_to_next')] = np.NaN
            # create feature: frequent in each year
            year_candidates = ['2014', '2015', '2016', '2017', '2018']
            for year in year_candidates:
                tmp['frequent_in_'+year] = len(tmp[tmp['year'] == year])
            # create feature: time shift for num_act
            num_act_shift_forward  = pd.concat([pd.Series(np.NaN), tmp['num_act'][:-1]])
            num_act_shift_forward.index = tmp.index
            tmp['num_act_shift1'] = num_act_shift_forward
            # create feature: time shift for num_teile
            num_teile_shift_forward  = pd.concat([pd.Series(np.NaN), tmp['num_teile'][:-1]])
            num_teile_shift_forward.index = tmp.index
            tmp['num_teile_shift1'] = num_teile_shift_forward
            # create feature: time shift for total aw
            total_aw_shift_forward  = pd.concat([pd.Series(np.NaN), tmp['total AW'][:-1]])
            total_aw_shift_forward.index = tmp.index
            tmp['total_aw_shift1'] = total_aw_shift_forward
            if len(tmp) > 2:
                num_act_shift_forward2 = pd.concat([pd.Series([np.NaN, np.NaN]), tmp['num_act'][:-2]])
                num_act_shift_forward2.index = tmp.index
                tmp['num_act_shift2'] = num_act_shift_forward2
                num_teile_shift_forward2 = pd.concat([pd.Series([np.NaN, np.NaN]), tmp['num_teile'][:-2]])
                num_teile_shift_forward2.index = tmp.index
                tmp['num_teile_shift2'] = num_teile_shift_forward2
                total_aw_shift_forward2 = pd.concat([pd.Series([np.NaN, np.NaN]), tmp['total AW'][:-2]])
                total_aw_shift_forward2.index = tmp.index
                tmp['total_aw_shift2'] = total_aw_shift_forward2
            else:
                tmp['num_act_shift2'] = np.NaN
                tmp['num_teile_shift2'] = np.NaN
                tmp['total_aw_shift2'] = np.NaN
            # create feature: repair age in year or in month
            tmp['first_date'] = tmp.iloc[0, tmp.columns.get_loc('Auftragsdatum')]
            tmp['repair_age'] = tmp['Auftragsdatum'] - tmp['first_date']
            tmp = tmp.drop('first_date', axis=1)
            # create feature: repair frequent in last X month
            time_distance_shift1 = pd.concat([pd.Series(np.NaN), tmp['time_distance_from_last'][:-1]])
            time_distance_shift1.index = tmp.index
            tmp['time_distance_shift1'] = time_distance_shift1
            time_distance_shift2 = pd.concat([pd.Series(np.NaN), tmp['time_distance_shift1'][:-1]])
            time_distance_shift2.index = tmp.index
            tmp['time_distance_shift2'] = time_distance_shift2
            time_distance_shift3 = pd.concat([pd.Series(np.NaN), tmp['time_distance_shift2'][:-1]])
            time_distance_shift3.index = tmp.index
            tmp['time_distance_shift3'] = time_distance_shift3
            tmp['time_distance_from_last2'] = tmp['time_distance_from_last'] + tmp['time_distance_shift1']
            tmp['time_distance_from_last3'] = tmp['time_distance_from_last2'] + tmp['time_distance_shift2']
            tmp['time_distance_from_last4'] = tmp['time_distance_from_last3'] + tmp['time_distance_shift3']
            def get_frequent(x, d = 30):
                if x['time_distance_from_last4'] < timedelta(days = d):
                    return 4
                elif x['time_distance_from_last3'] < timedelta(days = d):
                    return 3
                elif x['time_distance_from_last2'] < timedelta(days = d):
                    return 2
                elif x['time_distance_from_last'] < timedelta(days = d):
                    return 1
                else:
                    return 0
            tmp['frequent_in_last_1_month'] = tmp.apply(get_frequent, axis=1, args = [1*30])
            tmp['frequent_in_last_2_month'] = tmp.apply(get_frequent, axis=1, args = [2*30])
            tmp['frequent_in_last_3_month'] = tmp.apply(get_frequent, axis=1, args = [3*30])
            tmp['frequent_in_last_4_month'] = tmp.apply(get_frequent, axis=1, args = [4*30])
            tmp['frequent_in_last_5_month'] = tmp.apply(get_frequent, axis=1, args = [5*30])
            tmp['frequent_in_last_6_month'] = tmp.apply(get_frequent, axis=1, args = [6*30])
            # create feature: weather come in last X month
            tmp['weather_come_in_last_1_month'] = tmp['time_distance_from_last'] < timedelta(days = 1*30)
            tmp['weather_come_in_last_2_month'] = tmp['time_distance_from_last'] < timedelta(days = 2*30)
            tmp['weather_come_in_last_3_month'] = tmp['time_distance_from_last'] < timedelta(days = 3*30)
            tmp['weather_come_in_last_4_month'] = tmp['time_distance_from_last'] < timedelta(days = 4*30)
            tmp['weather_come_in_last_5_month'] = tmp['time_distance_from_last'] < timedelta(days = 5*30)
            tmp['weather_come_in_last_6_month'] = tmp['time_distance_from_last'] < timedelta(days = 6*30)
            # create feature: trend of frequent in the past
            def get_trend(x):
                xs = x[['time_distance_from_last', 'time_distance_shift1', 'time_distance_shift2', 'time_distance_shift3']].tolist()
                xs = [x.days for x in xs if not pd.isnull(x)]
                if len(xs) < 2:
                    return 0
                ys = range(1, len(xs) + 1)
                xm = sum(xs)/len(xs)
                ym = sum(ys)/len(ys)
                xy = [x*y for x, y in zip(xs, ys)]
                xx = [x*y for x, y in zip(xs, xs)]
                up = sum(xy) - len(xs)*xm*ym
                down = sum(xx) - len(xs)*xm*xm
                if down == 0:
                    return 0
                return up*1.0/down
            tmp['trend_of_frequent_in_the_past'] = tmp.apply(get_trend, axis=1)
            # mean km distance:
        else:
            # set features to NaN
            tmp['times'] = 1
            tmp['time_distance_from_last'] = pd.NaT
            tmp['time_distance_to_next'] = pd.NaT
            tmp['mean_time_distance_in_group'] = pd.NaT
            tmp['max_time_distance_in_group'] = pd.NaT
            tmp['min_time_distance_in_group'] = pd.NaT
            tmp['km_distance_from_last'] = np.NaN
            tmp['km_distance_to_next'] = np.NaN
            tmp['mean_km_distance_in_group'] = np.NaN
            tmp['max_km_distance_in_group'] = np.NaN
            tmp['min_km_distance_in_group'] = np.NaN
            tmp['frequent_in_2014'] = 1 if tmp.iloc[0, tmp.columns.get_loc('year')] == '2014' else 0
            tmp['frequent_in_2015'] = 1 if tmp.iloc[0, tmp.columns.get_loc('year')] == '2015' else 0
            tmp['frequent_in_2016'] = 1 if tmp.iloc[0, tmp.columns.get_loc('year')] == '2016' else 0
            tmp['frequent_in_2017'] = 1 if tmp.iloc[0, tmp.columns.get_loc('year')] == '2017' else 0
            tmp['frequent_in_2018'] = 1 if tmp.iloc[0, tmp.columns.get_loc('year')] == '2018' else 0
            tmp['num_act_shift1'] = np.NaN
            tmp['num_teile_shift1'] = np.NaN
            tmp['total_aw_shift1'] = np.NaN
            tmp['num_act_shift2'] = np.NaN
            tmp['num_teile_shift2'] = np.NaN
            tmp['total_aw_shift2'] = np.NaN
            tmp['time_distance_shift1'] = pd.NaT
            tmp['time_distance_shift2'] = pd.NaT
            tmp['time_distance_shift3'] = pd.NaT
            tmp['time_distance_from_last2'] = pd.NaT
            tmp['time_distance_from_last3'] = pd.NaT
            tmp['time_distance_from_last4'] = pd.NaT
            tmp['weather_come_in_last_1_month'] = False
            tmp['weather_come_in_last_2_month'] = False
            tmp['weather_come_in_last_3_month'] = False
            tmp['weather_come_in_last_4_month'] = False
            tmp['weather_come_in_last_5_month'] = False
            tmp['weather_come_in_last_6_month'] = False
            tmp['trend_of_frequent_in_the_past'] = 0
        out.append(tmp)   
    logger.info('finish creating features: time relvative')
    logger.info('saving temmporary dataframe')
    df3 = pd.concat(out)
    df3.to_pickle('df_after_feature_extraction.pkl') 
    logger.info('extract feature, weather a customer will come in X months for special Gruppe-Nr')
    tg = df3.copy()
    tg['weather_come_in_1_month'] = tg['time_distance_to_next'] < timedelta(days = 1*30)
    tg['weather_come_in_2_month'] = tg['time_distance_to_next'] < timedelta(days = 2*30)
    tg['weather_come_in_3_month'] = tg['time_distance_to_next'] < timedelta(days = 3*30)
    tg['weather_come_in_4_month'] = tg['time_distance_to_next'] < timedelta(days = 4*30)
    tg['weather_come_in_5_month'] = tg['time_distance_to_next'] < timedelta(days = 5*30)
    tg['weather_come_in_6_month'] = tg['time_distance_to_next'] < timedelta(days = 6*30)
    df3 = tg
    del tg
    gc.collect()
    logger.info('deal with NA value in df')
    nd = df.copy()
    # fill na with unknown
    fs_to_un = ['Markencode', 'Adressanredecode', 'Motorcode', 'Fahrzeugmodellnummer', 'Modell',
               'Typ', 'Getriebecode', 'Gewicht', 'Leistung (KW)', 'Erstzulassungsdatum']
    for col in fs_to_un:
        nd[col] = nd[col].fillna('unknown')
    nd['age_auto'] = nd['age_auto'].fillna(0)
    df = nd
    del nd
    gc.collect()
    loger.info('deal with NA value in df3')
    ndf = df3.copy()
    # drop times == 1
    ndf = ndf.drop(ndf[ndf['times'] == 1].index)
    # fill with 0
    fs_fill_0 = ['frequent_in_last_1_month', 'frequent_in_last_2_month', 'frequent_in_last_3_month', 
                'frequent_in_last_4_month', 'frequent_in_last_5_month', 'frequent_in_last_6_month',
                'num_act_shift1', 'num_act_shift2', 'num_teile_shift1', 'num_teile_shift2', 'total_aw_shift1',
                'total_aw_shift2']
    for col in fs_fill_0:
        ndf[col] = ndf[col].fillna(0)
    # fill with 0 timedelta
    ndf['repair_age'] = ndf['repair_age'].fillna(timedelta(0))
    # fill with mean: km_distance_from_last, km_distance_to_next
    ndf['km_distance_from_last'][ndf['km_distance_from_last'].isna()] = ndf['mean_km_distance_in_group'][ndf['km_distance_from_last'].isna()]
    ndf['km_distance_from_last'] = ndf['km_distance_from_last'].fillna(ndf['km_distance_from_last'].mean()) # fill the rest with total mean
    ndf['km_distance_to_next'][ndf['km_distance_to_next'].isna()] = ndf['mean_km_distance_in_group'][ndf['km_distance_to_next'].isna()]
    ndf['km_distance_to_next'] = ndf['km_distance_to_next'].fillna(ndf['km_distance_to_next'].mean())
    # fill with mean: time_distance_from_last
    # timedelta changed to int automatically, why???
    def set_f2_to_f1(x, feature1, feature2):
        #print(x['time_distance_from_last'])
        if type(x[feature1]) == type(pd.NaT):
            x[feature1] = x[feature2]
        return x
    ndf = ndf.apply(set_f2_to_f1, axis = 1, args = ['time_distance_from_last', 'mean_time_distance_in_group'])
    ndf = ndf.apply(set_f2_to_f1, axis = 1, args = ['time_distance_to_next', 'mean_time_distance_in_group'])
    # extract the feature, that we may will used in the training
    fs = ['Auftragsdatum', 'Fahrgestellnummer', 'frequent_in_2014', 'frequent_in_2015', 'frequent_in_2016',
         'frequent_in_2017', 'frequent_in_2018', 'frequent_in_last_1_month', 'frequent_in_last_2_month',
         'frequent_in_last_3_month', 'frequent_in_last_4_month', 'frequent_in_last_5_month', 'frequent_in_last_6_month',
         'km_distance_from_last', 'max_km_distance_in_group', 'min_km_distance_in_group', 'mean_km_distance_in_group',
         'mean_time_distance_in_group', 'max_time_distance_in_group', 'min_time_distance_in_group', 
         'num_act_shift1', 'num_act_shift2', 'num_teile_shift1', 'num_teile_shift2', 'repair_age',
         'time_distance_from_last', 'total_aw_shift1', 'total_aw_shift2', 'trend_of_frequent_in_the_past',
         'weather_come_in_last_1_month', 'weather_come_in_last_2_month', 'weather_come_in_last_3_month',
         'weather_come_in_last_4_month', 'weather_come_in_last_5_month', 'weather_come_in_last_6_month',
         'weather_come_in_1_month', 'weather_come_in_2_month', 'weather_come_in_3_month', 'weather_come_in_4_month',
         'weather_come_in_5_month', 'weather_come_in_6_month']
    ndf = ndf[fs]
    # remember that, we still have feature in df, so we have to merge the df3 back to df.
    # we name it df2 here for test
    df2 = pd.merge(df, ndf, on = ['Fahrgestellnummer', 'Auftragsdatum'], how = 'right')
    loger.info('deal with outlier in df and clip')
    ol = df2.copy()
    # Gewicht
    ol['Gewicht'][ol['Gewicht'] == 'unknown'] = '0,00'
    # Leistung (KW)
    ol['Leistung (KW)'][ol['Leistung (KW)'] == 'unknown']  = 0.0
    #df2.info()
    df2 = ol
    del ol
    gc.collect()
    loger.info('feature vor discretization')
    av = df2.copy()
    # Adressanredecode
    av['Adressanredecode'].loc[(av['Adressanredecode'] != 'Firma') & (av['Adressanredecode'] != 'Herr') & (av['Adressanredecode'] != 'Frau')] = 'other'
    # Motorcode: distribute according to the first Ziffer
    av['Motorcode'] = av['Motorcode'].map(lambda x: str(x).lower()[0] if x else x)
    # Motorcode: change the Motorcode which start with a number
    av['Motorcode'][av['Motorcode'].map(lambda x: True if re.search('[0-9\-]', x) else False)] = 'unknown'
    # Motorcode: change the Motorcode which account smaller than 100
    # combine items in Motor count < 100 will be set to other
    tmp = av.copy()
    tmp['count'] = 1
    motor = tmp[['Motorcode', 'count']]
    mc = motor.groupby('Motorcode', as_index = False).count()
    tmp = pd.merge(av, mc, how = 'left', on = 'Motorcode')
    tmp.loc[tmp['count'] < 100, 'Motorcode'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    av = tmp
    # Fahrzeugmodellnummer
    tmp = av.copy()
    tmp['count'] = 1
    motor = tmp[['Fahrzeugmodellnummer', 'count']]
    mc = motor.groupby('Fahrzeugmodellnummer', as_index = False).count()
    tmp = pd.merge(av, mc, how = 'left', on = 'Fahrzeugmodellnummer')
    tmp.loc[tmp['count'] < 100, 'Fahrzeugmodellnummer'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    av = tmp
    # modell nummer
    tmp = av.copy()
    tmp['count'] = 1
    motor = tmp[['Modell', 'count']]
    mc = motor.groupby('Modell', as_index = False).count()
    tmp = pd.merge(av, mc, how = 'left', on = 'Modell')
    tmp.loc[tmp['count'] < 100, 'Modell'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    av = tmp
    # getriebecode
    tmp = av.copy()
    tmp['count'] = 1
    motor = tmp[['Getriebecode', 'count']]
    mc = motor.groupby('Getriebecode', as_index = False).count()
    tmp = pd.merge(av, mc, how = 'left', on = 'Getriebecode')
    tmp.loc[tmp['count'] < 100, 'Getriebecode'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    av = tmp
    # Typ
    tmp = av.copy()
    tmp['count'] = 1
    motor = tmp[['Typ', 'count']]
    mc = motor.groupby('Typ', as_index = False).count()
    tmp = pd.merge(av, mc, how = 'left', on = 'Typ')
    tmp.loc[tmp['count'] < 100, 'Typ'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    av = tmp
    df2 = av
    del av
    gc.collect()
    loger.info('encoding')
    ec = df2.copy()
    features_to_ordinal = ['Markencode', 'Fahrzeugmodellnummer', 'Modell', 'Getriebecode']
    ordinal_encoder = OrdinalEncoder()
    # select faetures
    sf = ec[features_to_ordinal]
    # name
    name_ordinal_encode = [f + '_ordinal_encode' for f in features_to_ordinal]
    # fit and transform
    ordinal_encoder.fit(sf)
    tmp = ordinal_encoder.transform(sf)
    # create Dataframe
    tmp = pd.DataFrame(tmp, columns = name_ordinal_encode)
    # combine
    ec = pd.concat([ec, tmp], axis = 1)
    # del and gc
    del sf, tmp, features_to_ordinal, ordinal_encoder
    gc.collect()
    features_to_onehot = ['Lagerortcode', 'Adressanredecode', 'Motorcode', 'age_autohaus', 'Typ']
    onehot_encoder = OneHotEncoder()
    for feature in features_to_onehot:
        sf = ec[feature]
        tmp = onehot_encoder.fit_transform(sf.values.reshape(-1, 1)).toarray()
        _, l = tmp.shape
        columns = [feature + '_onehot_' + str(i) for i in range(l)]
        tmp = pd.DataFrame(tmp, columns = columns)
        ec = pd.concat([ec, tmp], axis = 1)
    # del and gc
    del sf, tmp, columns
    gc.collect()
    df2 = ec
    del ec
    gc.collect()
    loger.info('discretization')
    fc = df2.copy()
    # Gewicht
    fc['Gewicht'] = fc['Gewicht'].map(lambda x: re.sub(',', '.', x))
    # binning, number of bin???????
    # use onehot to do the encode, would be better???
    # when n_bins larger als 3, it throws ecxeption: bins must be monotonically increasing or decreasing ???
    n_bins = 2
    bin_width = KBinsDiscretizer(n_bins = n_bins, encode = 'onehot', strategy = 'uniform')
    bin_deep = KBinsDiscretizer(n_bins = n_bins, encode = 'onehot', strategy = 'quantile')
    bin_kmeans = KBinsDiscretizer(n_bins = n_bins, encode = 'onehot', strategy = 'kmeans')
    # features to do the binning
    features_to_binning = ['num_act', 'total AW', 'KM-Stand', 'Gewicht', 'Leistung (KW)', 'age_auto',
                           'Markencode_ordinal_encode', 'Fahrzeugmodellnummer_ordinal_encode', 'Modell_ordinal_encode']
    # bining with ordinal encoder
    # select faetures
    tmp = fc[features_to_binning]
    # set name
    name_width = [f + '_bin_width' for f in features_to_binning]
    name_deep = [f + '_bin_deep' for f in features_to_binning]
    name_kmeans = [f + '_bin_kmeans' for f in features_to_binning]
    # fit and transform
    bin_width.fit(tmp)
    bin_deep.fit(tmp)
    bin_kmeans.fit(tmp)
    tmp_w = bin_width.transform(tmp)
    tmp_d = bin_deep.transform(tmp)
    tmp_k = bin_kmeans.transform(tmp)
    # columns names
    cols_name_width = []
    cols_name_deep = []
    cols_name_kmean = []
    for feature in features_to_binning:
        for i in range(n_bins):
            cols_name_width.append(feature + '_onehot_bin_width_' + str(i))
            cols_name_deep.append(feature + '_onehot_bin_deep_' + str(i))
            cols_name_kmean.append(feature + '_onehot_bin_kmean_'+ str(i))
    # transform to Dataframe
    tmp_w = pd.DataFrame(tmp_w.toarray(), columns = cols_name_width)
    tmp_d = pd.DataFrame(tmp_d.toarray(), columns = cols_name_deep)
    tmp_k = pd.DataFrame(tmp_k.toarray(), columns = cols_name_kmean)
    # concatenate
    fc = pd.concat([fc, tmp_w, tmp_d, tmp_k], axis = 1)
    # del and gc
    del tmp, tmp_w, tmp_d, tmp_k
    gc.collect()
    features_to_mdlp = ['num_act', 'total AW', 'KM-Stand', 'Gewicht', 'Leistung (KW)', 'age_auto', 
                    'Typ_ordinal_encode', 'Fahrzeugmodellnummer_ordinal_encode', 'Modell_ordinal_encode',
                    'num_teile', 'month', 'year', 'day']
    label = 'weather_come_in_last_4_month'
    out_path_data, out_path_bins, return_bins, class_label, features = None, None, False, None, None
    features = features_to_mdlp
    class_label = label
    discretizer = MDLP_Discretizer(dataset=fc, class_label=class_label, features=features, out_path_data=out_path_data, out_path_bins=out_path_bins)
    tmp = discretizer.apply_cutpoints(out_data_path=out_path_data, out_bins_path=out_path_bins)
    tmp.to_pickle('discretizer.pkl')
    loger.info('end')

class ABDict:
    """
    make sure that only two columns in the df, otherwise only the first two columns will be used
    用于两列数据互补空值
    """
    def __init__(self, path, header = True, sep = ';'):
        self.dic = self._load_dict(path, header, sep) 
        self.dic_conv = self._load_dict_conv(path, header, sep)
        print(len(self.dic))
        print(len(self.dic_conv))
        counter = 0
        ########
        # test #
        ########
        #for i in self.dic.keys():
        #    if counter > 1:
        #        break
        #    print(i, self.dic[i])
        #    counter += 1
        #print('schraube' in self.dic_conv.keys())
    
    def AToB(self, a):
        if a in self.dic.keys():
            out = self.dic[a][0]
            if type(out) == list:
                print(out)
            return out
        else:
            #print(a)
            return float('nan')
    
    def BToA(self, b):
        if b in self.dic_conv.keys():
            out = self.dic_conv[b][0]
            #print(out)
            return out
        else:
            #print(b)
            return float('nan')

    def _load_dict(self, path, header, sep):
        dic = {} 
        with open(path) as fi:
            counter = 0
            for li in fi:
                if header and counter ==0:
                    # drop first line if header is true
                    counter += 1
                    continue
                li = li.strip()   # 忘记去换行符了，导致调了一晚上的错
                items = li.split(sep)
                if items[0] not in dic.keys():
                    # if not exit, add new item
                    if items[1] != 'nan':
                        dic[items[0]] = [items[1]]
                elif items[1] not in dic[items[0]]:
                    # if item exit but value not eixt, add the value to the list
                    dic[items[0]].append(items[1])
                counter += 1
        return dic
    
    def _load_dict_conv(self, path, header, sep):
        dic = {}
        with open(path) as fi:
            counter = 0
            for li in fi:
                if header and counter == 0:
                    # drop first line if header is true
                    counter += 1
                    continue
                li = li.strip()
                items = li.split(sep)
                if items[1] not in dic.keys():
                    if items[0] != 'nan':
                        #print(items[0])
                        dic[items[1]] = [items[0]]
                elif items[0] not in dic[items[1]]:
                    dic[items[1]].append(items[0])
                counter += 1
        return dic

# 转化为auftrag table， 但是这次合并的是Teile-Nr项
# 给的数据的每一行都是一个维修项，初衷是，把属于同一个auftrag的维修项合并到一起，看一下，在同一个Auftrag中，经常一起修的是那些内容

def toAuftragTable(df, att, auftn, clean = True):
    """
    input:
        df, DataFrame:
            the dataframe
        att, string:
            the column name of the target attribute
        auftn, string:
            the column name of the aftragsnummer attribute
        clean:
            when true, drop the null item in auftn attribute.
    output:
        df_g, DataFrame:
            dataframe contrains two columns auftn and att
            type of item in att is string, separate with ';'
    """
    # assert: make sure the type of the attributes inputted
    
    # extract the att and date columns
    df = df[[att, auftn]]
    # set type to object
    #df[att] = df[att].astype('object')
    #df[auftn] = df[auftn].astype('object')
    # if clean is True, drop the fake data, like the null data
    if clean:
        print("Falls Null date exist, drop these dates directly")
        #df = df.drop(df[df[att].isnull()].index)
        df = df.drop(df[df[auftn].isnull()].index)
    # group and sum 
    df_g = df.groupby([auftn], as_index = False).apply(agg)
    return df_g

# apply 只能对单行进行处理，而不是对整个分组进行处理，所以估计应该把axis换成1，比较好
def agg(x):
    # 是否用‘ ’分隔会比较好，这样就不用对初始的属性，
    # x 在这里是dataframe？？？
    #x = [str(i) for i in x]
    x = x.apply(lambda x: ';'.join(set([str(i) for i in x])), axis = 0)
    #x = x.apply(lambda x: ' '.join(set(x)), axis = 0)
    #print(x.columns.values)
    return x

if __name__ == '__main__':
    main()