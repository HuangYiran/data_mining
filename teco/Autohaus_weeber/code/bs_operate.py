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
sys.path.append('/Users/ihuangyiran/Documents/Workplace_Python/data_mining/utils/dict')
import os
import time
import warnings

# operate
from pattern.de import parse, conjugate, singularize, pluralize

# others
import argparse
from FreDict import FreDict
from IdxDict import IdxDict

parser = argparse.ArgumentParser()
parser.add_argument('-data', default = 'data/Autohaus_weeber/leonberg_werkstattposten_2014.csv', help = 'the path of the target dataset')

def bs_operate(li):
    """
    input string:
        a 'describe' attribute
    """
    # to string
    li = str(li).lower()
    # recover: problem with '.' in string
    #dict_re = {'ae': 'ä', 'oe': 'ö', 'ue': 'ü', 'ß': 'ss', 'u.': 'und ', 'fzg': 'fahrzeug', ' f ': ' für '}
    dict_re = {'ae': 'ä', 'oe': 'ö', 'ue': 'ü', 'ß': 'ss', 'fzg': 'fahrzeug', ' f ': ' '}
    for i, j in dict_re.items():
        if i in li:
            li = re.sub(i, j, li)
    #print(li,0)
    # replace punctuation with ' '
    li = re.sub("[\s+\.\!_,$%^*(+\"\')-:]", " ", li)
    #print(li,1)
    # remove the number in the string
    li = re.sub('\d', '', li)
    #print(li,2)
    # remove word with one ziffer
    li = re.sub(' [a-zA-Z] ', "", li)
    #print(li,3)
    # plural to odd
    words = li.strip().split(' ')
    #words_sin = [pluralize(word.strip()) if len(word) > 0 and parse(word).split('/')[1] == 'NN' and parse(word).split('/')[1] != 'IN' else word.strip() for word in words]
    words_sin = [pluralize(word.decode('utf-8').strip()) for word in words if len(word)> 0 and  parse(word.decode('utf-8')).split('/')[1] != 'IN' ]
    li = ' '.join(words_sin)
    #print(li,4)
    # upper
    li = li.upper()
    #print(li,5)
    return li

def toAuftragTable(df, att, auftn, clean = True):
    """
    input:
        df, DataFrame:
            the dataframe
        att, string:
            the column name of the target attribute
        auftn, string:
            the column name of the aftragsnummer attribute
    output:
        df_g, DataFrame:
            dataframe contrains two columns auftn and att
            type of item in att is string, separate with ';'
    """
    # assert: make sure the type of the attributes inputted

    # extract the att and date columns
    df = df[[att, auftn]]
    # if clean is True, drop the fake data, like the null data
    if clean:
        print("Null date exist, drop these dates directly")
        df = df.drop(df[df[att].isnull()].index)
        df = df.drop(df[df[auftn].isnull()].index)
    # group and sum
    df_g = df.groupby([auftn], as_index = False).apply(agg)
    return df_g

# apply 只能对单行进行处理，而不是对整个分组进行处理，所以估计应该把axis换成1，比较好
def agg(x):
    # 是否用‘ ’分隔会比较好，这样就不用对初始的属性，
    #x = x.apply(lambda x: ';'.join(set(x)), axis = 0)
    x = x.apply(lambda x: ' '.join(set(x)), axis = 0)
    #print(x.columns.values)
    return x

def run_fp(df):
    df.to_csv('/tmp/tmp.csv', sep = ';', index = False)
    lis = [li.rstrip('\n')[1:-1] for li in open('/tmp/tmp.csv')]
    lin = [li.split(',') for li in lis]
    idx = []
    for li in lin:
        tmp = []
        for i in li:
            tmp.append(int(i.strip()))
        idx.append(tmp)
    # run the fp algm
    result = []
    for itemset, support in find_frequent_itemsets(idx, 10, True):
        result.append((itemset,support))
    result = sorted(result, key=lambda i: i[0])
    print('#'*100)
    for itemset, support in result:
        print(str(itemset) + ' ' + str(support))

def main():
    opt = parser.parse_args()
    att = 'Beschreibung'
    df = pd.read_csv(opt.data, sep = ';')
    bs = df[['Auftragsnummer', att]]
    # operate bs and extract attribute 1
    bs[att+'_1'] = bs[att].map(bs_operate)
    # extract attribute2
    bs[att+'_1'].to_csv('/tmp/tmp.csv', sep = ';', index = False)
    fdic = FreDict('/tmp/tmp.csv', header = True, sep = ';')
    bs[att+'_2'] = bs[att+'_1'].map(fdic.get_best)
    # translate the table to Auftrag table
    tmp1 = toAuftragTable(bs, att+'_1', auftn = 'Auftragsnummer', clean = True)
    tmp2 = toAuftragTable(bs, att+'_2', auftn = 'Auftragsnummer', clean = True)
    # create idxDicts
    tmp1[att+'_1'].to_csv('/tmp/tmp.csv', sep = ';', index = False)
    idxDict1 = IdxDict('/tmp/tmp.csv', header = False)
    tmp2[att+'_2'].to_csv('/tmp/tmp.csv', sep = ';', index = False)
    idxDict2 = IdxDict('/tmp/tmp.csv', header = False)
    # transform word to index
    tmp1[att+'_1'] = tmp1[att+'_1'].map(idxDict1.li_to_idx)
    tmp2[att+'_2'] = tmp2[att+'_2'].map(idxDict2.li_to_idx)
    # run fp-algorithm
    run_fp(tmp[att+'_1'])
    run_fp(tmp[att+'_2'])

if __name__ == '__main__':
    main()
