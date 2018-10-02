import numpy as np


def show_tools():
    print("""
    d_train = pd.read_csv('dir')
    get the number of null wert with:
        d_train.isnull().sum()
    
    For missing data, we always have two methods: complete it(pd.DataFrame.fillna) or delete it(pd.DataFrame.drop)
        d_train['Age'].fillna(d_train['Age'].median(), inplace = True) # numeric wert
        d_train['Embarked'].fillna(d_train['Age'].mode()[0], inplace = True) # object wert
        d_train.drop(['PassengerId', 'Cabin', 'Ticket'], axis = 1, inplace = True)
    
    One problem of the code above is that the value we used is only affected by the feature itseilf. Sametime we may want to use a similar feature to help to fill the value. For example when we want to fill the null value in Fare feature, we can distribute the entries with pclass and fill the value with the median in each group.
    besides we can also use the self defined methods:
    fillna_with_friends()
    fillna_with_group(df, tg, groupby = None, agg = None)
    fillna_with_mean_and_random_noise(df, fe)
    """)

def fillna_with_friends():
    """
    solve the problem mentioned above
    1. calculate the correlation with friend features, choose the feature with witch correlation larger than a threshold
    2. each these feature have a vote for the fill value, we use the mode value to fill the na.
    """
    #TODO
    pass

def fillna_with_group(df, tg, groupby = None, agg = None):
    """
    fill na iwth group agg value
    先copy一下数据集，然后用gb和agg建新列，最后填补na值，删新列，返回数据集
    input:
        df: pd.DataFrame
            data frame
        tg: String
            target column
        groupby: String
            column to do the groupby
        agg: String
            name of the agg function
    """
    #TODO 
    pass

def fillna_with_mean_and_random_noise(df, fe):
    """
    input:
        df: dataframe
        fe: type of string, target feature
    """
    # make sure the feature is type of int or float
    #TODO assert
    avg = df[fe].mean()
    std = df[fe].std()
    null_count = df[fe].isnull().sum()
    null_random_list = np.random.randint(avg-std, avg+std, size = null_count)
    df[fe][np.isnan(df[fe])] = null_random_list
    #df.loc[df[fe].isnull(), fe] = null_random_list
    #df[fe] = df[fe].astype(int)
