import pandas as pd
import matplotlib.pyplot as plt

import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix, fbeta_score, f1_score
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

rf = RandomForestClassifier()

grid_space={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200],
              'max_features':[1,3,5,7],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[1,2,3]
           }

grid = GridSearchCV(rf, param_grid=grid_space, cv=3, scoring='accuracy')
# Y_train = Y_train.values.ravel()
model_grid = grid.fit(X_train, Y_train.values.ravel()) 

print(model_grid.best_score_)