# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math

from tqdm import tqdm

def show_tools():
    print("""
    introduce some useful methods:
    downcast_dtypes(df)
    cal_cond_entropy(df, fs, tg)
    """)

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int16)

    return df

def cal_cond_entropy(df, fs, tg):
    """
    calculate the conditional entropy of the target feature

    input:
        df: DataFrame
            the inputted data frame
        fs: list of String
            the features to calcute the entropy
        tg: String
            the target String
    """
    #TODO  assert
    len_data = len(df)
    for col in tqdm(fs):
        data_col = df[[col, tg]]
        data_col['tmp'] = 0
        # calculate p(x)
        col_count_x = data_col.groupby(col)[tg].count().reset_index().rename(columns = {tg: 'ratio_x'})
        col_count_x['ratio_x'] = col_count_x['ratio_x']/len_data*1.0
        col_all = pd.merge(data_col, col_count_x, on = col, how = 'left')
        # calculate p(y,x)
        col_count_xy = data_col.groupby([col, tg])['tmp'].count().reset_index().rename(columns = {'tmp': 'ratio_xy'})
        col_count_xy['ratio_xy'] = col_count_xy['ratio_xy']/len_data*1.0
        col_all = pd.merge(col_all, col_count_xy, on = [col, tg], how = 'left')
        # calculate p(y|x)
        col_all['ratio_y/x'] = col_all['ratio_xy']/col_all['ratio_x']
        print(col_all)

        t_enp = 0
        for index in range(len(col_all)):
            t_enp += col_all.loc[index, 'ratio_xy'] * math.log(col_all.loc[index, 'ratio_y/x']) * -1.0
        print('Condition Entropy of feature %s is: %f'%(col, t_enp))

def cal_cond_entropy_pairwise(df, fs, tg):
    """
    pairwise
    """
    #TODO
    pass
