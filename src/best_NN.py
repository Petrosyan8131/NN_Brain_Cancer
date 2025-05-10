import pandas as pd
import matplotlib.pyplot as plt

import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import GridSearchCV

# импорт и подготовка датасета
df = pd.read_csv('data/scaled_train.csv')

rs = {'Прогрессия'}
tr = set(df.columns)-rs

X = df[list(tr)]
Y = df[list(rs)]

def randomoversample(x, y):
    ros = RandomOverSampler(random_state=10)
    return ros.fit_resample(x, y)

def smote(x, y):
    sm =  SMOTE(random_state=10, k_neighbors=5)
    return sm.fit_resample(x, y)

X_res, Y_res = smote(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2)
data = X_train, X_test, Y_train, Y_test

# def f2_func(y_true, y_pred):
#     f2_score = fbeta_score(y_true, y_pred, beta=2)
#     return f2_score

# def my_f2_scorer():
#     return make_scorer(f2_func)

f2_scorer = make_scorer(fbeta_score, beta=2)

def RandForestparams(x_train, y_train, score):
    rf = RandomForestClassifier()

    grid_space={'max_depth':[3,5,10,None],
                'n_estimators':[10,100],
                'max_features':[1,3,5,7],
                'min_samples_leaf':[1,2,3],
                'min_samples_split':[1,2,3]
            }

    grid = GridSearchCV(rf, param_grid=grid_space, cv=3, scoring=score)
    # Y_train = Y_train.values.ravel()
    model_grid = grid.fit(x_train, y_train.values.ravel()) 
    return model_grid

def DecTreeparams(x_train, y_train, score):
    dt = DecisionTreeClassifier(random_state=0)
    
    grid_space={'max_depth':[3,5,10,None],
                'min_samples_split':[2, 4, 6],  
                'min_samples_leaf':[1, 3, 5],
                'min_impurity_decrease':[0.0, 0.3],
                'max_features':[1,3,5,7],
                'min_weight_fraction_leaf':[0.0, 0.5],
                'max_leaf_nodes':[2,20]
                }
    grid = GridSearchCV(dt, param_grid=grid_space, cv=3, scoring=score)
    model_grid = grid.fit(x_train, y_train)
    return model_grid

def Logregparams(x_train, y_train, score):
    lr = LogisticRegression()
    
    grid_space = {'tol':[0.0001, 0.01],
                  'C':[1.0, 5.0],
                  'intercept_scaling':[1.0, 4.0],
                  'max_iter':[100, 200, 300],
                  'verbose':[0, 2, 4, 7]   
            }
    grid = GridSearchCV(lr, param_grid=grid_space, cv=3, scoring=score)
    y_train = y_train.values.ravel()
    model_grid = grid.fit(x_train, y_train)
    return model_grid

def KNNparams(x_train, y_train, score):
    model_neighbor = KNeighborsClassifier()
    
    grid_space = {'n_neighbors':[5, 7, 12, 18],
                   'leaf_size':[15, 30, 50, 60, 70],
                   'p':[2, 5, 7, 10]
                }
    grid = GridSearchCV(model_neighbor, param_grid=grid_space, cv=3, scoring=score)
    y_train = y_train.values.ravel()
    model_grid = grid.fit(x_train, y_train)
    return model_grid

def SVCparams(x_train, y_train, score):
    clf = SVC(kernel='rbf',)
    
    grid_space = {'C':[0.0, 2.0],
                  'degree':[3, 4, 6, 8],
                  'gamma':[0.4, 1.5],
                  'coef0':[0.0, 0.8],
                  'tol':[0.001, 0.1],
                  'cache_size':[200, 300],
                  'max_iter':[100, 200, 300]
            }
    grid = GridSearchCV(clf, param_grid=grid_space, cv=3, scoring=score)
    y_train = y_train.values.ravel()
    model_grid = grid.fit(x_train, y_train)
    return model_grid

forestmodel_grid = RandForestparams(X_train, Y_train, f2_scorer)
treemodel_grid = DecTreeparams(X_train, Y_train, f2_scorer)
logrecmodel_grid = Logregparams(X_train, Y_train, f2_scorer)
knnmodel_grid = KNNparams(X_train, Y_train, f2_scorer)
svcmodel_grid = SVCparams(X_train, Y_train, f2_scorer)

print("-"*70)
print('For RandomForest:')
print('Best hyperparameters are: '+str(forestmodel_grid.best_params_))
print('Best score is: '+str(forestmodel_grid.best_score_))
print("-"*70)
print('For DecisionTree:')
print('Best hyperparameters are: '+str(treemodel_grid.best_params_))
print('Best score is: '+str(treemodel_grid.best_score_))
print("-"*70)
print('For LogisticRegression:')
print('Best hyperparameters are: '+str(logrecmodel_grid.best_params_))
print('Best score is: '+str(logrecmodel_grid.best_score_))
print("-"*70)
print('For KNN:')
print('Best hyperparameters are: '+str(knnmodel_grid.best_params_))
print('Best score is: '+str(knnmodel_grid.best_score_))
print("-"*70)
print('For SVC:')
print('Best hyperparameters are: '+str(svcmodel_grid.best_params_))
print('Best score is: '+str(svcmodel_grid.best_score_))
print("-"*70)
