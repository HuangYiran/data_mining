#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn import model_selection
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
        tune_model = finetune_gridsearch(model = 'adaBoost')
        tune_model.fit(train_x, train_y)
        print('After DT Parameters: ', tune_model.best_params_)
	model_best = tune_model.best_estimator_
	print('best score {}'.format(tune_model.best_score_))
        tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100
        tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100
        the models that we can use in finetune_gridsearch function include:
        adaBoost: AdaBoostClassifier
        bagging: BaggingClassifier # not fs(feature selection)
        extraTrees: ExtraTreesClassifier
        gradientBoosting: GradientBoostingClassifier
        randomForest: RandomForestClassifier
        gaussianProcess: GaussianProcessClassifier # not fs
        logisticRegression: LogisticRegressionCV
        bernoulliNB: BernoulliNB
        kNeighbors: KNeighborsClassifier # not fs
        svc: SVC
        xgbc: XGBClassifier

        we can plot the learning curve with the function
        plot_learning_curve(estimator, title, X, y, ylim = None, cv = None, n_jobs = -1, train_sizes = np.linspace(.1,1.0, 5))

        7. besides we can also use feature selection to tune the model
        es_rfe = feature_selection.RFECV(estimator, step = 1, scoring = 'accuracy', cv = cv_split)
        es_rfe.fit(train_x, train_y)
        X_rfe = train_x.columns.values[es_rfe.get_support()] # extract features
        rfe_result = model_selection.cross_validation(dtree, train_x[X_rfe], train_y, cv =cv_split)
        8. why choose one model, when we have more
        vote_est = [
            ('ada', ensemble.AdaBoostClassifier()),
            ('bc', ensemble.BaggingClassifier()),
            ...
        ]
        vote_hard = ensemble.VotingClassifier(estimators = vote_est, voting = 'hard')
        vote_soft = ensemble.VotingClassifier(estimators = vote_est, voting = 'soft')

        9. to submit the result we use following codes:
        Y_test = vote_soft.predict(X_test)
        test_Survived = pd.Series(Y_test, name = 'Survived')
        PassengerID = d_test['PassengerId']
        submit = pd.concat([PassengerID, test_Survived], axis = 1)
        submit.to_csv('../submit.csv', index = False)
        ps. 一个问题是，这种投票机制，无法实现为各个子model私人定制训练属性。因此只能先超参选后，最后加一个总的属性选，要克服这个问题，应该自己重写一个ensemble的方法，实现用训练好的模型进行投票。(为完成)
    """)

def finetune_gridsearch(model = 'ada'):
    """
    ada: AdaBoostClassifier
    """
    param_grid, mod = _get_param_grid_and_model_for_gridsearch(model)
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
    tune_model = model_selection.GridSearchCV(mod, param_grid = param_grid, scoring = 'roc_auc', cv = cv_split)
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
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
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


def run_models(x_train, y_train, mla, x_test = None):
    """
    run algrithm in the mla list, and save the result to table
    input:
        x_train: train data - X
        y_train: training data - Target
        mla: list of machine learning algorithms
	x_test: if set, predict with the x_test set instead of x_train
    output:
        MLA_compare: pandas dataframe
        MLA_predict_pd: prediction of the x_train set(or x_test set if setted)
    """
    MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
    MLA_predict = {}
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
	if x_test is not None:
	    MLA_predict[MLA_name] = alg.predict(x_test)
	else:
	    MLA_predict[MLA_name] = alg.predict(x_train)
        row_index += 1
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    MLA_predict_pd = pd.DataFrame.from_dict(MLA_predict)
    return MLA_compare, MLA_predict_pd

def run_models_with_finetune(x_train, y_train, mla, x_test = None):
    """
    same function as run_models but here for each model, we run finetune_gridsearch() first
    """
    name_mla = []
    for alg in mla:
        name_mla.append(alg.__class__.__name__)
    finetune_mla = []
    for alg in name_mla:
        tune_model = finetune__gridsearch(alg)
        tune_model.fit(x_train, y_train)
        model_best = tune_model.best_estimator_
        finetune_mla.append(model_best)
    MLA_compare, MLA_predict = run_models(x_train, y_train, finetune_mla, x_test)
    return MLA_compare, MLA_predict

####################
## assist functions
####################
def _get_param_grid_and_model_for_gridsearch(model = 'ada'):
    """
    input:
        model: type of model
    output:
        param_grid, type of dict
        mod: MLA model 
    """
    grid_n_estimator = [10, 50, 100, 300]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.0001, .001, .01, .03, .05, .1, .25]
    grid_max_depth = [2, 4, 6, 8, 10, None]
    grid_min_samples = [5, 10, .03, .05, 10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]

    if model == 'adaBoost' or model == 'AdaBoostClassifier':
        grid_param = {
#                'best_estimator__criterion': ['gini', 'entropy'],
#                'best_estimator__splitter': ['best', 'random'],
                'n_estimators': grid_n_estimator, # default = 50
                'learning_rate': grid_learn, # default = 1
                'algorithm': ['SAMME', 'SAMME.R'], # default = 'SAMME.R'
                'random_state': grid_seed
                }
        mod = ensemble.AdaBoostClassifier()
    elif model == 'bagging' or model == 'BaggingClassifier':
        grid_param = {
                'n_estimators': grid_n_estimator, # default = 10
                'max_samples': grid_ratio, # default = 1.0
                'random_state': grid_seed
                }
        mod = ensemble.BaggingClassifier()
    elif model == 'extraTrees' or model == 'ExtraTreesClassifier':
        grid_param = {
                'n_estimators': grid_n_estimator, # default  = 10
                'criterion': grid_criterion, # default = 'gini'
                'max_depth': grid_max_depth, # default = None
                'max_features': [1, 3, 7],
		'min_samples_split': [2, 3, 7],
                'min_samples_leaf': [1, 3, 7],
                'bootstrap': [False],
                'random_state': grid_seed
                }
        mod = ensemble.ExtraTreesClassifier()
    elif model == 'gradientBoosting' or model == 'GradientBoostingClassifier':
        grid_param = {
                'loss': ['deviance', 'exponential'], # default = 'deviance'
                'learning_rate': [.05], # default = 0.1
                'n_estimators': [300], # default = 100
                'criterion': ['friedman_mse', 'mse', 'mae'], # default = 'friedman_mse'
                'max_depth': grid_max_depth, # default = 3
                'min_samples_leaf': [100,150],
                'max_features': [.3, .1],
                'random_state': grid_seed
                }
        mod = ensemble.GradientBoostingClassifier()
    elif model == 'randomForest' or model == 'RandomForestClassifier':
        grid_param = {
                'n_estimators': grid_n_estimator, # default = 0
                'criterion': grid_criterion, # default = 'gini'
                'max_depth': grid_max_depth, # default = None
                'oob_score': [True], # default = False
                'random_state': grid_seed,
                'min_samples_split': [2, 3, 7],
                'min_samples_leaf': [1, 3, 7],
                }
        mod = ensemble.RandomForestClassifier()
    elif model == 'gaussianProcess' or model == 'GaussianProcessClassifier':
        grid_param = {
                'max_iter_predict': grid_n_estimator, # default = 100
                'random_state': grid_seed
                }
        mod = gaussian_process.GaussianProcessClassifier()
    elif model == 'decisionTree' or model == 'DecisionTreeClassifier':
        param_grid = {
                'criterion': ['gini', 'entropy'], # default gini
                'splitter': ['best', 'random'], # default best
                'max_depth': [2, 4, 6, 8, 10, None], # default None
                'min_samples_split': [2, 5, 10, .03, .05], # minimum subset size before new split, default 2
                'min_samples_lear': [1, 5, 10, .03, .05], # minimum subset size after new split, default 1
                'max_features': [None, 'auto'], # max features to consider when performing split; default noen or all
                'random_state': [0] # seed or control random number generator
                }
        mod = tree.DecisionTreeClassifier()
    elif model == 'logisticRegression' or model == 'LogisticRegressionCV':
        grid_param = {
                'fit_intercept': grid_bool, # default = True
                #'penalty': ['11', '12'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], # default: lbfgs
                'random_state': grid_seed
                }
        mod = linear_model.LogisticRegressionCV() # ??
    elif model == 'bernoulliNB' or model == 'BernoulliNB':
        grid_param = {
                'alpha': grid_ratio, # default1.0
                }
        mod = naive_bayes.BernoulliNB()
    elif model == 'kNeighbors' or model == 'KNeighborsClassifier':
        grid_param = {
                'n_neighbors': [1, 2, 3, 4, 5, 6, 7], # default = 5
                'weights': ['uniform', 'distance'], #default = 'uniform'
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
        mod = neighbors.KNeighborsClassifier()
    elif model == 'svc' or model == 'SVC':
        grid_param = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [1, 2, 3, 4, 5], # default = 1.0
                'gamma': grid_ratio, # default = auto
                'decision_function_shape': ['ovo', 'ovr'], # default = ovr
                'probability': [True],
                'random_state': grid_seed
                }
        mod = svm.SVC(probability = True)
    elif model == 'xgbc' or model == 'XGBClassifier':
        grid_param = {
                'learning_rate': grid_learn, # default = .3
                'max_depth': [1, 2, 4, 6, 8, 10], # default = 2
                'n_estimators': grid_n_estimator,
                'seed': grid_seed
                }
        mod = XGBClassifier()
    else:
        print('unrecognized model: '+ model)
        grid_param = {}
        mod = None
    return grid_param, mod



