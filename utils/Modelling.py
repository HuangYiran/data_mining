#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn import model_selection
from sklearn import tree, random_forest, ExtraTreesClassifier
from sklearn.model_selection import learning_curve, Shuffle_split
from sklearn.ensemble import random_forest, GradientBoostingClassifier
from sklearn.svm import SVC


"""
Modeling 有几个注意点
0. 选择baseline
1. 使用什么算法，建议是尽可能多的进行尝试，并不存在golden truth的说
2. corss validation，为防止过拟合，这是必不可少的
3. table输出，当然也可以画图，这是为了方便对比统计
"""

def show_tools():
    print("""
        1. firstly we need to select and initialize models:
        MLA = [...]
        2. cross validation
        cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
        3. create table to compare MLA metrics
        MLA_columns = ['MLA Name', 'MLA PArameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
        4. index through MLA and save performance to table
        MLA_compare.loc[row_index, 'MLA Name'] = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
        cv_results = model_selection.cross_validate(alg, train_x, train_y, cv = cv_split)
        MLA_compare.loc[row_index, 'MLA Time'] = cv.results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv.results['train_score'].mean() # out of date!!!!
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv.results['test_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv.result['tst_score'].std()*3
        5. next step is valuate the result, usually we use confusion matrix to do it
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        classes = ['Dead', 'Survived']
        plt.figure()
        plot_confusion_matrix(cm, classes, normalize=False, title='..', cmap=plt.cm.Blues)
        6. to get a better score, we need to fine tune the model
        tune_model = finetune_dt_gridsearch()
        tune_model.fit(train_x, train_y)
        print('After DT Parameters: ', tune_model.best_params_)
        tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100
        tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100
        similar we have these methods:
        finetune_etc_gridsearch() # ExtraTreesClassifier
        finetune_gbc_gridsearch() # GradientBoostingClassifier
        finetune_svc_gridsearch() 
        finetune_rf_gridsearch() # random forest
        we can plot the learning curve with the function
        plot_learning_curve(estimator, title, X, y, ylim = None, cv = None, n_jobs = -1, train_sizes = np.linspace(.1,1.0, 5))
        7. besides we can also use feature selection to tune the model
        es_rfe = feature_selectoin.RFECV(estimator, step = 1, scoring = 'accuracy', cv = cv_split)
        es_rfe.fit(train_x, train_y)
        X_rfe = train_x.columns.values[es_rfe.get_support()] # extract features
        rfe_result = model_selection.cross_validation(dtree, train_x[X_rfe], train_y, cv =cv_split)
    """)


def finetune_dt_gridsearch():
    """
    wrap dt model with gridSearchCV
    特定MLA的参数取值范围都是类似的，所以写这类方法主要是为了偷懒，主要内容就是设定param_grid
    output:
        out: type of model_selection.GridSearchCV
    """
    param_grid = {
            'criterion': ['gini', 'entropy'], # default gini
            'splitter': ['best', 'random'], # default best
            'max_depth': [2, 4, 6, 8, 10, None], # default None
            'min_samples_split': [2, 5, 10, .03, .05], # minimum subset size before new split, default 2
            'min_samples_lear': [1, 5, 10, .03, .05], # minimum subset size after new split, default 1
            'max_features': [None, 'auto'], # max features to consider when performing split; default noen or all
            'random_state': [0] # seed or control random number generator
            }
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
    print(list(model_selection.ParameterGrid(param_grid)))
    tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid = param_grid,
            scoring = 'roc_auc', cv = cv_split)
    return tune_model

def finetune_extc_gridsearch():
    param_grid = {
            'max_depth': [None],
            'max_features': [1,3,7],
            'min_samples_split': [2,3,7],
            'min_samples_lear': [1,3,7],
            'bootstrap': [False],
            'n_estimators': [300,600],
            'criterion': ['gini']
            }
    #cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
    kfold = model_selection.StratifiedKFold(n_splits = 10)
    tune_model = GridSearchCV(ExtraTreesClassifier(), param_grid = param_grid, cv = kfold, scoring = 'accuracy', verbose = 1)
    return tune_model

def finetune_gbc_gridsearch():
    GBC = GradientBoostingClassifier()
    param_grid = {
            'loss': ['deviance'],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.1, 0.05, 0.01],
            'max_depth': [4, 8],
            'min_samples_leaf': [100, 150],
            'max_features': [0.3, 0.1]
            }
    kfold = model_selection.StratifiedKFold(n_splits = 10)
    tune_model = GridSearchCV(GBC, param_grid = param_grid, cv = kfold, scoring = 'accuracy', verbose = 1)
    return tune_model

def finetune_svc_gridsearch():
    SVMC = SVC(probability = True)
    param_grid = {
            'kernel': ['rbf'],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C': [1, 10, 50, 100, 200, 300, 1000]
            }
    kfold = model_selection.StratifiedKFold(n_splits = 10)
    tune_model = GridSearch(SVMC, param_grid = param_grid, cv = kfold, scoring = 'accuracy', verbose = 1)
    return tune_model

def finetune_rf_gridsearch():
    """
    random forest
    """
    param_grid = {
            'max_depth': [None],
            'max_features': [1,3,7],
            'min_samples_split': [2,3,7],
            'min_samples_leaf': [1,3,7],
            'bootstrap': [False],
            'n_estimators': [300, 600],
            'criterion': ['gini']
            }
    kfold = model_selection.StratifiedKFold(n_splits = 10)
    tune_model = GridSearchCV(random_forest, param_grid = param_grid, cv = kfold, scoring = 'accuracy', verbose = 1)
    return tune_model

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    """
    plot confusion matrix
    input:
        cm: confusion matrix
        classes: target class
        normalize: need normalize or not
        title: name of the plot
        cmap: color section
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis)]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arrange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() /2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt),
                horizontalalignment = 'center',
                color='white' if cm[i,j]>thresh else 'black')

def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None, n_jobs = -1, train_sizes = np.linspace(.1,1.0,5)):
    """
    Generate a simple plot of the test and training learning curve
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.mean(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.mean(test_scores, axis = 1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = 'g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = 'Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = 'Cross-validation score')
    plt.legend(loc = 'best')
    return plt


def run_models(x_train, y_train, mla):
    """
    run algrithm in the mla list, and save the result to table
    input:
        x_train: train data - X
        y_train: training data - Target
        mla: list of machine learning algorithms
    output:
        out: pandas dataframe
    """
    MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
    
    row_index = 0
    for alg in mla:
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
        # score model 
        cv_results = model_selection.cross_validate(alg, x_train, y_train, cv = cv_split)
        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3
        row_index += 1
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    return MLA_compare
