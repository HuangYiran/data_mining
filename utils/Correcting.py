import numpy as np
from collections import Counter
from tqdm import tqdm

def show_tools():
    print("""
    for numeric feature we can use this method:
        detect_outliers_with_IQR(df, n, features) # n is the min number of outlier
        detect_outliers_with_IQR_groupby(df, n , features, gb = None)
        detect_outliers_with_object_account(df, n, features)
    for the outlier we can either delete it or fill a new value as in Complete. We make the decision according to the final loss.
        train=train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
    """)

def detect_outliers_with_IQR(df, n, features):
    """
    numeric outliers detection
    steps from Yassine Ghouzam
    input:
        n Int:
            only more than n columns say an item is an outlier, it is a outlier
    """
    outlier_indices = []
    # iterate over feature(columns)
    for col in features:
        # 1st quartile 
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile
        Q3 = np.percentile(df[col], 75)
        # nterquartile rankge (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col]<Q1-outlier_step)|(df[col]>Q3+outlier_step)].index
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v>n)
    return multiple_outliers

def detect_outliers_with_IQR_groupby(df, n , features, gb = None):
    """
    detect_outliers_with_IQR_groupby
    """
    if gb is None:
        outliers = detect_outliers_with_IQR(df, n, features)
    else:
        outliers = []
        tg = df[gb].drop_duplicates().tolist()
        for i in tqdm(tg):
            chunk = df[df[gb]==i]
            tmp_o = detect_outliers_with_IQR(chunk, n, features)
            outliers.extend(tmp_o)
        return outliers

def detect_outliers_with_object_account(df, n, features):
    """
    calculate the account of each class, base on these account find the outlier
    """
    #TODO
    pass
