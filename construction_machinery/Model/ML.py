import numpy as np
from tqdm.auto import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

##Model classification
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def train_val_test(x_train, y_train, x_val, y_val, x_test, model, param_grid):
    # Parameter
    param_list = list(ParameterGrid(param_grid))
    val_score = []

    # Grid Search
    for p in tqdm(param_list, desc='Hyperparameter Search...'):
        clf = model(**p)
        clf.fit(x_train, y_train)
        val_score.append(f1_score(y_val, clf.predict(x_val), average='macro'))

    # Test Performance
    best_clf = model(**param_list[np.argmax(val_score)])
    best_clf.fit(x_train, y_train)

    # Test Performance
    val_f1 = f1_score(y_val, best_clf.predict(x_val), average='macro')
    y_pred = best_clf.predict(x_test)

    return val_f1, y_pred

# SVM Classifier
def SVM(x_train, y_train, x_val, y_val, x_test):
    # SVM Param
    param_grid = {'gamma': ['scale'], 'random_state': [42], 'probability': [True],
                  'C': [0.01, 0.1, 1], 'kernel': ['poly', 'linear', 'rbf']}

    # Check Performance
    val_f1, y_pred = train_val_test(x_train, y_train, x_val, y_val, x_test, SVC, param_grid)
    return val_f1, y_pred

# XGB Classifier
def XGB(x_train, y_train, x_val, y_val, x_test):
    # XGB Param
    param_grid = {'eval_metric': ['mlogloss'], 'random_state': [6], 'max_depth': [5, 7],
                   'subsample': [0.6, 0.8], 'eta': [0.2], 'lambda': [0.1, 0.3], 'alpha': [0]}

    # Check Performance
    val_f1, y_pred = train_val_test(x_train, y_train, x_val, y_val, x_test,
                                    xgb.XGBClassifier, param_grid)
    return val_f1, y_pred

# Logistic Regression
def Logistic_Regression(x_train, y_train, x_val, y_val, x_test):
    # Logistic Regression Param
    param_grid = {'random_state': [42], 'penalty': ['l1', 'l2'],
                  'C': np.logspace(-5, 0, 10), 'solver': ['liblinear']}

    # Check Performance
    val_f1, y_pred = train_val_test(x_train, y_train, x_val, y_val, x_test,
                                    LogisticRegression, param_grid)
    return val_f1, y_pred

# Random Forest
def Random_Forest(x_train, y_train, x_val, y_val, x_test):
    # Random Forest Param
    param_grid = {'random_state': [4], 'criterion': ['gini', 'entropy'],
                  'max_depth': [3, 5, 7]}

    # Check Performance
    val_f1, y_pred = train_val_test(x_train, y_train, x_val, y_val, x_test,
                                    RandomForestClassifier, param_grid)
    return val_f1, y_pred

# Decision Tree
def Decision_Tree(x_train, y_train, x_val, y_val, x_test):
    # Random Forest Param
    param_grid = {'random_state': [10], 'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                  'max_depth': [3, 5, 7]}

    # Check Performance
    val_f1, y_pred = train_val_test(x_train, y_train, x_val, y_val, x_test,
                                    DecisionTreeClassifier, param_grid)
    return val_f1, y_pred

# Machine Learning All
def MachineLearning_Performance(x_train, y_train, x_val, y_val, x_test):
    result_dict = {'DT': None, 'LR': None, 'RF': None, 'SVM': None, 'XGB': None}
    function_list = [SVM, XGB, Logistic_Regression, Random_Forest, Decision_Tree]

    for i, m in enumerate(['SVM', 'XGB', 'LR', 'RF', 'DT']):
        print('Model: {}'.format(m))
        val_f1, y_pred = function_list[i](x_train, y_train, x_val, y_val, x_test)
        result_dict[m] = {'F1': val_f1, 'pred': y_pred}

    return result_dict