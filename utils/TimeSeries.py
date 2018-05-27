# -*- coding: utf-8 -*-

import pandas as pd
import gc


from tqdm import tqdm
from itertools import product
from sklearn.model_selection import KFold
from pandas.tseries.holiday import USFederalHolidayCalendar

def show_tools():
    """
    list the tools included in this package
    """
    print("
            List the tools available in this package:
            create_date_features(df, date) #TODO
            create_grid(df, keys, target) #TODO
            create_lag_features_with_time_feature(df, cols, time, n = 5, fillna = True)
            create_lag_features_ohen_time_feature(df, cols, n = 5, fillna = True)
            mean_encoder(df, cols, tg)
            ")

def create_date_features(df = None, date = None):
    """
    create date features for time series data
    input:
        df: the dataframe 
        date: the data feature
    output:
        df
    """
    #TODO

def create_date_feature_bisiness_quater(df = None, date = None):
    """
    create data feature: bisiness quater ????
    """
    #TODO

def create_date_feature_daytime(df = None, date = None):
    """
    create date features day time for time series data
    """
    df[date] = pd.to_datetime(df[date])
    df['dayOfWeek'] = df[date].dt.dayofweek
    df['dayOfMonth'] = df[date].dt.day #???
    df['year'] = df[date].dt.year
    df['month'] = df[date].dt.month

def create_date_feature_is_public_holiday(df = None, date = None, country = 'US'):
    """
    create date feature: is holiday
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2014-01-01', end='2014-12-31').to_pydatetime()
    if datetime.datetime(2014,01,01) in holidays:
        print True
    """
    #TODO

def create_date_feature_is_month_end(df = None, date = None):
    """
    create date features: is month end
    """
    #TODO

def create_date_feature_is_weekend(df = None, date = None):
    """
    create date feature: is weekend
    """
    #TODO

def create_date_feature_is_weekday(df = None, date = None):
    """
    create date feature: is weekday
    """
    #TODO

def create_date_feature_season(df = None, date = None):
    """
    create date feature: season
    """
    #TODO

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

def create_lag_features_with_time_feature(df = None, cols = None, time = None, n = 5, fillna = True):
    """
    create lag features for time series
    input:
        df: data frame
        cols: columns to creat lag feature, type of list
        time: the feature that determine the time, can be day, month other year, type of string
        n: number of lag to create, type of list or int
        fillna: weather fillna or not, it will affect the na value in other feature
    output:
        df
    """
    # assert
    assert(df is not None and cols is not None)
    # set attributes
    cols_to_rename = cols
    print(cols_to_rename)
    if type(n) == list:
        shift_range = n
    elif type(n) == int:
        shift_range = range(1, n+1)
    else:
        print("type of n is flase, set it to default: 5")
        shift_range = range(1, 6)
    # try to get the new features
    for month_shift in tqdm(shift_range):
        train_shift = df.copy()
        train_shift[time] = train_shift[time] + month_shift
        foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
        train_shift = train_shift.rename(columns=foo)
        df = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)
    del train_shift
    gc.collect()
    return df

def create_lag_features_ohen_time_feature(df = None, cols = None, n = 5, fillna = True):
    """
    create lag feature, before creating the features we need to sort the dataframe with time:
    input:
        df: the data frame
        cols: columns to create the lag features
        n: number of lag, type of int or list
        fillna: weather fill na or not, it will not affect the na value in other features
    output:
        df
    """
    # assert
    assert(df is not None and cols is not None)
    # set feature
    cols_to_rename = cols
    print(cols_to_rename)
    if type(n) == list:
        shift_range = n
    elif type(n) == int:
        shfit_range = range(1, n+1)
    else:
        print("type of n is false, set it to default: 5")
        shift_range = range(1, n+1)
    for month_shift in tqdm(shift_range):
        train_shift = df[cols_to_rename]
        train_shift = train_shift.shift(month_shift).fillna(0)
        foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
        train_shift = train_shift.rename(columns = foo)
        df = pd.concat([df, train_shift], axis = 1)
    del train_shift
    gc.collect()
    return df

def create_window_feature(df, cols = None, col2 = None, win_size = 2, win_type = None, min_periods = 1, agg = 'mean'):
    """
    create window feature for time series
    agg include: sum, mean, var, std, min, max, corr, cov, skew, kurt
    input:
        df: Datafrmae
        cols: columns to create window feature, type of list
        win_size: the size of the windows
        win_type: the type of the window
        min_periods: mininum count of the window to calcute the agg
        agg: aggregate function 
    """
    foo = lambda x: x+'_win_'+win_type+'_'+agg
    f_agg = getattr(pd.core.window.Rolling, agg)
    tmp = None
    if agg == 'corr' or agg == 'corr_cov':
        assert(cols2 is not None)
        if len(cols.columns) > 1:
            print("only accept one target col, the result now is unpredictable for me, hehehe")
        tmp = f_agg(pd.DataFrame.rolling(df[cols], window = win_size, min_periods = min_periods, win_type = win_type), pd.DataFrame.rolling(df[col2], window = win_size, min_periods = min_periods, win_type = win_type))
    else:
        tmp = f_agg(pd.DataFrame.rolling(df[cols], window = win_size, min_periods = min_periods, win_type = win_type))
    if tmp is None:
        print("unrecoginze agg function, return the original df")
    else:
        tmp = tmp.rename(columns = foo)
        df = pd.concat([df, tmp], axis = 1)
    return df

def mean_encoder(df, cols = None, tg = None):
    """
    calculate the mean encoder for high cardinality features,
    most of the code is copied from the code of anqitu
    input:
        df: dataframe, type of pd.DataFrame
        cols: columns to calcute the mean encoder, type of list
        tg: the target that used to calcute the mean encoder, type of list
    output:
        train(/df)
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
    return train
