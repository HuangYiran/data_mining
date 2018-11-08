#! -*- coding: utf-8 -*-

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
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/data_mining/utils/')
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/data_mining/utils/dict')
import os
import time
import warnings

# operate
from pattern.de import parse, conjugate, singularize, pluralize

# others
from FreDict import FreDict
from IdxDict import IdxDict

parser = argparse.ArgumentParser()
parser.add_argument('-data', default = '../data/Autohaus_weeber/leonberg_werkstattposten_2014.csv', help = 'the path of the target dataset')
parser.add_argument('-att', default = 'Anrede', help = 'The attribute to calculate the frequent')

def toFreqTable(df, att, date, clean = True):
    """
    input:
        df, DataFrame:
            the dataframe
        att, string:
            the column name of the target attribute
        date, string:
            the column name of the date attribute
    """
    # extract the att and date columns
    df = df[[att, date]]
    
    # change string date to date type
    # detect outlier
    fake = df[date].map(check_date)
    # if clean is True, drop the fake data
    if clean:
        print("Fake date exist, drop these dates directly")
        df = df.drop(df[fake].index)
    df[date] = pd.to_datetime(df[date])
    # add new column named 'count'
    df['count'] = pd.Series([1 for i in range(len(df))])
    # group and sum 
    df_g = df.groupby([date, att], as_index = False).agg({'count': 'sum'})
    return df_g

def check_date(s, sep = '.'):
    """
    only check the date grobly,
    set the fake date to True and the right date to false
    """
    items = s.split('.')
    #print(items)
    if len(items) <3:
        return True
    day = int(items[0])
    month = int(items[1])
    year = int(items[2])
    if day <= 31 and day > 0 and month <= 12 and month > 0 and year > 0:
        return False
    else:
        return True





def main():
    opt = parser.parse_args()
    att_t = opt.att
    att_d = 'Auftragsnummer'
    data = opt.read_csv(opt.data, sep = ';')
    # create frequent table but still lack of empty data
    df_ref = toFreqTable(data, att_t, att_d)
    # create new table 
    def create(x):
        tmp = pd.DataFrame({att_t: df_ref[att_t].unique()})
        tmp[att_d] = x.iloc[0, 0]
        return tmp
    df_new = df_ref.groupby(att_d, as_index = False).apply(create)
    # merge the new table to the frequent table
    df_new = df_new.reset_index()[[att_d, att_t]]
    df_full = pd.merge(df_ref, df_new, how = 'right')
    # change NaN to 0
    tmp['count'] = tmp.iloc[:, 2].replace(np.nan,0)
    # extract the year and month info
    tmp['month'] = tmp['Auftragsdatum'].map(lambda x: str(x)[:7])
    # aggregate the data in month
    df_agg = tmp.groupby(['date', 'Adressanredecode'], as_index = False).sum()





if __name__ == '__main__':
    main()
