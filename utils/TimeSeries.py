# -*- coding: utf-8 -*-

import pandas as pd


from tqdm import tqdm
from itertools import product
from sklearn.model_selection import KFold


def create_grid(df = None, keys = None , target = None):
    """
    作用不明，暂时跳过
    input:
        df: data frae type of pd
        keys: keys items type of list
        target: target item type of string
    """
    #TODO
    # assert
    #assert(df is not None and key is not None and target is not None)
    # generate the gird
    #grid = []
    #for block_num in sale_train['date_block_num'].unique():
    #    cur_shops = sale_train[sale_train['date_block_num']==block_num]['shop_id'].unique()
    #    cur_items = sale_train[sale_train['date_block_num']==block_num]['item_id'].unique()
    #    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

def mean_encoder(df, cols = None, tg = None):
    """
    calculate the mean encoder for high cardinality features,
    most of the code is copied from the code of anqitu
    input:
        df: dataframe, type of pd.DataFrame
        cols: columns to calcute the mean encoder, type of list
        tg: the target that used to calcute the mean encoder, type of list
    """
    # set attribute
    train = df
    Target = tg
    mean_encoded_col = cols
    SEED = 0
    global_mean =  train[Target].mean()
    
    # get mean_encoder
    for col in tqdm(mean_encoded_col):
        col_tr = train[[col] + [Target]]
        corrcoefs = pd.DataFrame(columns = ['Cor'])

        # 3.1.1 Mean encodings - KFold scheme
        kf = KFold(n_splits = 5, shuffle = False, random_state = SEED)

        col_tr[col + '_cnt_month_mean_Kfold'] = global_mean
        for tr_ind, val_ind in kf.split(col_tr):
            X_tr, X_val = col_tr.iloc[tr_ind], col_tr.iloc[val_ind]
            means = X_val[col].map(X_tr.groupby(col)[Target].mean())
            X_val[col + '_cnt_month_mean_Kfold'] = means
            col_tr.iloc[val_ind] = X_val
            # X_val.head()
        col_tr.fillna(global_mean, inplace = True)
        corrcoefs.loc[col + '_cnt_month_mean_Kfold'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Kfold'])[0][1]

        # 3.1.2 Mean encodings - Leave-one-out scheme
        item_id_target_sum = col_tr.groupby(col)[Target].sum()
        item_id_target_count = col_tr.groupby(col)[Target].count()
        col_tr[col + '_cnt_month_sum'] = col_tr[col].map(item_id_target_sum)
        col_tr[col + '_cnt_month_count'] = col_tr[col].map(item_id_target_count)
        col_tr[col + '_target_mean_LOO'] = (col_tr[col + '_cnt_month_sum'] - col_tr[Target]) / (col_tr[col + '_cnt_month_count'] - 1)
        col_tr.fillna(global_mean, inplace = True)
        corrcoefs.loc[col + '_target_mean_LOO'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_LOO'])[0][1]


        # 3.1.3 Mean encodings - Smoothing
        item_id_target_mean = col_tr.groupby(col)[Target].mean()
        item_id_target_count = col_tr.groupby(col)[Target].count()
        col_tr[col + '_cnt_month_mean'] = col_tr[col].map(item_id_target_mean)
        col_tr[col + '_cnt_month_count'] = col_tr[col].map(item_id_target_count)
        alpha = 100
        col_tr[col + '_cnt_month_mean_Smooth'] = (col_tr[col + '_cnt_month_mean'] *  col_tr[col + '_cnt_month_count'] + global_mean * alpha) / (alpha + col_tr[col + '_cnt_month_count'])
        col_tr[col + '_cnt_month_mean_Smooth'].fillna(global_mean, inplace=True)
        corrcoefs.loc[col + '_cnt_month_mean_Smooth'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Smooth'])[0][1]


        # 3.1.4 Mean encodings - Expanding mean scheme
        cumsum = col_tr.groupby(col)[Target].cumsum() - col_tr[Target]
        sumcnt = col_tr.groupby(col).cumcount()
        col_tr[col + '_cnt_month_mean_Expanding'] = cumsum / sumcnt
        col_tr[col + '_cnt_month_mean_Expanding'].fillna(global_mean, inplace=True)
        corrcoefs.loc[col + '_cnt_month_mean_Expanding'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Expanding'])[0][1]

        train = pd.concat([train, col_tr[corrcoefs['Cor'].idxmax()]], axis = 1)
        print(corrcoefs.sort_values('Cor'))
