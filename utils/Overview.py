#!-*- coding: utf8 -*-
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

def show_tools():
    print("""
    following is same methods that often used in overlook step
    d_train = pd.read_csv('dir')
    d_train.info() # show number of total entries, features type and number
                   # we should code the object features and fill the null value
    d_train.sample(10)
    d_train.head(10)
    d_train.tail(10)
    d_train.describe() # list basic statistic value for numeric feature
    d_train.describe(include = 'all')

    besides we can also plot the feature the get a better viewing, for a viry first look, we always instrested in the distibution and target rate
    import matplotlib.pyplot as plt
    import seaborn as sns
    f, ax = plt.subplot(3,4,figsize = (20,16))
    # count the number with countplot
    sns.countplot('Pclass', data = d_train, ax = ax[0,0])
    # view the target rate by adding a third dimension hue
    sns.countplot('SibSp', hue = 'Survived', data=train, ax = ax[0,3], palette = 'hus1')
    sns.boxplot(x='Pclass', y= 'Age', data = train, ax = ax[0,2])
    # we can also view the target rate by drawing to figure in one place
    sns.distplot(d_train[d_train['Survival']==0]['Age'].dropna(), ax=ax[1,0],kde=False,color='r',bins = 5)
    sns.distplot(d_train[d_train['Survival']==1]['Age'].dropna(), ax=ax[1,0],kde=False,color-'g',bins = 5)
    # use dist for continue value
    sns.distplot(d_train['Fare'].dropna(), ax=ax[2,0],kde=Flase, color='b')
    # for the target rate of two features we can use swarmplot
    sns.swarmplot(x='Pclass',y='Fare',hue='Survived', data=d_train, ax=ax[2,1])
    as[0,3].set_title('Survival Rate by SibSp')
    ax[0,2].set_title('Age Box Plot by class')

    after this phase, you should have a overview of all the features, and determine which feature need 4C.
    ps: 当数据量比较足的时候，应该把数据分成多个部分进行分析，只有当多部分分析结果相同的属性，才进行保留，否则，可能是噪音
    """)

def basic_plots(df, tg, fs='all', cartesian = False):
    """
    a very first look into the data, show the distribution and target rate of each features
    if necessary, plot the distribution and target rate for the pairs in the features
    input:
        df: the dataframe
        tg: target features
        fs: the features we concern
        cartesian: weather plot for the pairs or not 
    methods:
        according to the type and number of unique value of the feature, choose suited figure to plot the data
    problems:
        并不很了解plot的机制，所以不知道是否可以这样调用
    """
    #TODO
    pass


