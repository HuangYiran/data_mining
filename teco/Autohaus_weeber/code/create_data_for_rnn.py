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


def main():
    # read files
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
    #data1 = pd.concat([d1,d2,d3,d4], 0)
    d6 = std_wsp_2014.copy()
    d7 = std_wsp_2015.copy()
    d8 = std_wsp_2016.copy()
    d9 = std_wsp_2017.copy()
    d10 = std_wsp_2018.copy()
    # for training we use data from 14 - 17 in lb_wsp
    train = pd.concat([d1,d2,d3,d4], 0)
    test = d5
    data = [train, test]
    df = train.copy()
    # remove the items, that contain ziffer in the AW-Nr, set data type to int
    df['AW-Nr'][df['AW-Nr'].isnull()] = 0 # set null value to 0
    df['AW-Nr'][df['AW-Nr'] == '99999999'] = 0 # set 99999999 to 0
    df['AW-Nr'] = df['AW-Nr'].map(lambda x: re.sub(' ', '', str(x)))# remove space between the number
    del_index = df[df['AW-Nr'].map(lambda x: bool(re.search('[a-zA-Z]', str(x))))].index # remove fake data
    df = df.drop(del_index) 
    df['AW-Nr'] = df['AW-Nr'].astype('float')
    # remove wierd auftragsnummer
    df = df.drop(df[df['Auftragsnummer'] == '103K'].index)
    df = df.drop(df[df['Auftragsnummer'] == '77KW'].index)
    # transform to auftrag table and summieren the AW-Nr attribute for the same auftrag
    df_auf = toAuftragTable(df, 'AW-Nr', 'Auftragsnummer', clean = True)
    # sum the AW-Nr
    df_auf['AW-Nr'] = df_auf['AW-Nr'].map(lambda x: sum([float(i) for i in x.split(';')]))
    # merget the data original dataframe to df
    df_sel = df[['Auftragsnummer', 'KM-Stand', 'Auftragsdatum', 'Markencode', 'Lagerortcode', 'Adressanredecode', 'Fahrgestellnummer', 'Motorcode', 'Modell', 'Getriebecode']]
    df_sel = df_sel.drop_duplicates()
    df_auf = pd.merge(df_auf, df_sel, how = 'left', on = 'Auftragsnummer')
    # extract month and year attribute
    df_auf['month'] = df_auf['Auftragsdatum'].map(lambda x: int(x.split('.')[1]))
    df_auf['year'] = df_auf['Auftragsdatum'].map(lambda x: int(x.split('.')[2]))
    # combine items in Markencode
    df_auf.loc[(df_auf['Markencode'] != 'VW') & (df_auf['Markencode'] != 'AUDI') & (df_auf['Markencode'] != 'SKODA'), 'Markencode'] = 'others'
    # !!!!!!! 可以包装成一个方法，给属性中满足特定条件的项换值
    # combine items in Motor count < 100 will be set to other
    tmp = df_auf.copy()
    tmp['count'] = 1
    motor = tmp[['Motorcode', 'count']]
    mc = motor.groupby('Motorcode', as_index = False).count()
    tmp = pd.merge(df_auf, mc, how = 'left', on = 'Motorcode')
    tmp.loc[tmp['count'] < 100, 'Motorcode'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    df_auf = tmp
    # modell nummer
    tmp = df_auf.copy()
    tmp['count'] = 1
    motor = tmp[['Modell', 'count']]
    mc = motor.groupby('Modell', as_index = False).count()
    tmp = pd.merge(df_auf, mc, how = 'left', on = 'Modell')
    tmp.loc[tmp['count'] < 100, 'Modell'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    df_auf = tmp
    # getriebecode
    tmp = df_auf.copy()
    tmp['count'] = 1
    motor = tmp[['Getriebecode', 'count']]
    mc = motor.groupby('Getriebecode', as_index = False).count()
    tmp = pd.merge(df_auf, mc, how = 'left', on = 'Getriebecode')
    tmp.loc[tmp['count'] < 100, 'Getriebecode'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    df_auf = tmp
    # Adressanredecode
    tmp = df_auf.copy()
    tmp['count'] = 1
    motor = tmp[['Adressanredecode', 'count']]
    mc = motor.groupby('Adressanredecode', as_index = False).count()
    tmp = pd.merge(df_auf, mc, how = 'left', on = 'Adressanredecode')
    tmp.loc[tmp['count'] < 400, 'Adressanredecode'] = 'others'
    tmp.drop('count', axis= 1, inplace= True)
    df_auf = tmp
    # groupby fahrgestellsnummer and compute the date distance between the nearby items
    # the condition of the operate below is: we sort the data with 'Fahrgestellnumm' and 'Auftragsdatum'
    tmp = df_auf.copy()
    td = tmp[['Auftragsdatum', 'Fahrgestellnummer']]
    #td['Auftragsdatum'] = pd.to_datetime(td['Auftragsdatum'])
    td = td.drop_duplicates() # drop duplicate
    td['Auftragsdatum1'] = pd.to_datetime(td['Auftragsdatum'], dayfirst= True) # transfer to datetime
    td = td.sort_values(['Fahrgestellnummer', 'Auftragsdatum1']) # sort with Fahrgestellnummer
    td = td.reset_index(drop= True) # shift the data so that we can compute the distance
    td['Auftragsdatum2'] = pd.concat([pd.Series(datetime(2010,1,1)), td['Auftragsdatum1'][:-1]],ignore_index=True)
    #td['Auftragsdatum2'] = pd.concat([td['Auftragsdatum'][1:], pd.Series('0.0.0')],ignore_index=True)
    # compute the date distance
    #td['date_dis'] = pd.to_datetime(td['Auftragsdatum']) - pd.to_datetime(td['Auftragsdatum2'])
    td['date_dis'] = td['Auftragsdatum1'] - td['Auftragsdatum2']
    #td = td[['Auftragsdatum', 'Fahrgestellnummer', 'date_dis']]
    # merge back to the

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
        print("Null date exist, drop these dates directly")
        #df = df.drop(df[df[att].isnull()].index)
        df = df.drop(df[df[auftn].isnull()].index)
    # group and sum 
    df_g = df.groupby([auftn], as_index = False).apply(agg)
    return df_g

# apply 只能对单行进行处理，而不是对整个分组进行处理，所以估计应该把axis换成1，比较好
def agg(x):
    # 是否用‘ ’分隔会比较好，这样就不用对初始的属性，
    # x 在这里是dataframe？？？，还是就是一列
    #x = [str(i) for i in x]
    x = x.apply(lambda x: ';'.join(set([str(i) for i in x])), axis = 0)
    #x = x.apply(lambda x: ' '.join(set(x)), axis = 0)
    #print(x.columns.values)
    return x

if __name__ == '__main__':
    main()