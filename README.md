This Repository is used to collect(write) some useful methods that can be used in data mining task.
All this function is stored under the dir utils. These methods include:
##### Prepare
introduce some useful package that can be used in the process of data mining 
- jupyter notebook
- import pandas as pd
- import matplotlib
- import matplotlib.pyplot as plt
- import matplotlib.pylab as pylab
- import seaborn as sns
- import numpy as np
- import scipy as sp  # scientific computing and advance mathematics
- import sklearn  # machine learning algorithm
- from sklearn import smv, tree, linear_model, neighbors, naive_byaes, ensemble, discriminant_analysis, gaussian_process
- from sgboost import XGBClassifier
- from sklearn import feature_selection
- from sklearn import model_selection
- from sklearn import metrics

##### Common
- downcast_dtypes(df)
- cal_cond_entropy(df, fs, tg)

##### Overview
- d_train = pd.read_csv('dir')
- d_train.info() # show number of total entries, features type and number
- d_train.sample(10)
- d_train.head(10)
- d_train.tail(10)
- d_train.describe() # list basic statistic value for numeric feature
- d_train.describe(include = 'all')

##### Completing
- d_train.isnull().sum()
- d_train['Age'].fillna(d_train['Age'].median(), inplace = True) # numeric wert
- d_train['Embarked'].fillna(d_train['Age'].mode()[0], inplace = True) # object wert
- d_train.drop(['PassengerId', 'Cabin', 'Ticket'], axis = 1, inplace = True)
- fillna_with_friends(df)
- fillna_with_group(df, tg, groupby = None, agg = None)
- fillna_with_mean_and_random_noise(df, fe)

##### Correcting
- detect_outliers_with_IQR(df, n, features) # n is the min number of outlier
- detect_outliers_with_IQR_groupby(df, n , features, gb = None)
- detect_outliers_with_object_account(df, n, features)
- train=train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

##### Converting
- LabelEncoder
```
  from sklearn.preprocessing import LabelEncoder
  label = LabelEncoder()
  label.fit_transform(dataset['Sex'])
```
- show_kde(df, tg, fs)
- show_skew(df)
- fix_skew(df, fs)
- classify_with_clauster(df, tg, fs)
- show_corr_heatmap(df, fs)
- show_pairplots(df, fs)

##### Creating
- create_basic_statistic_groupby(df, fs, groupby = None)

##### Modelling(only show self define functions here)
- finetune_gridsearch(model = 'ada')
- plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues)
- plot_learning_curve(estimator, title, X, y, ylim = None, cv = None, n_jobs = -1, train_sizes = np.linspace(.1,1.0,5))
- run_models(x_train, y_train, mla, x_test = None)
- run_models_with_finetune(x_train, y_train, mla, x_test = None)

##### TimeSeries
- create_date_features(df, date) #TODO
- create_date_feature_bisiness_quater(df = None, date = None)
- create_date_feature_daytime(df = None, date = None)
- create_date_feature_is_public_holiday(df, date, start, end, country = 'US')
- create_date_feature_is_month_end(df = None, date = None, last = 1)
- create_date_feature_is_weekend(df = None, date = None)
- create_date_feature_is_weekday(df = None, date = None)
- create_date_feature_season(df = None, date = None)
- create_grid(df, keys, target) #TODO
- create_lag_features_with_time_feature(df, cols, time, n = 5, fillna = True)
- create_lag_features_ohne_time_feature(df, cols, n = 5, fillna = True)
- create_window_feature(df, cols = None, col2 = None, win_size = 2, win_type = None, min_periods = 1, agg = 'mean')
- mean_encoder(df, cols, tg)
