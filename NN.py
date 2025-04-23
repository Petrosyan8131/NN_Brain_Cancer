import pandas as pd
import matplotlib.pyplot as plt

import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay, confusion_matrix, fbeta_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE

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

# используемые модели
def KNN(data):
    data_train, data_test, res_train, res_test = data
    model_neighbor = KNeighborsClassifier(n_neighbors=5)
    res_train = res_train.values.ravel()
    model_neighbor = model_neighbor.fit(data_train, res_train)
    return model_neighbor

def SVM(data):
    data_train, data_test, res_train, res_test = data
    clf = SVC(kernel='rbf')
    res_train = res_train.values.ravel()
    clf = clf.fit(data_train, res_train)
    return clf

def DecisionTree(data):
    data_train, data_test, res_train, res_test = data
    dt = DecisionTreeClassifier(random_state=0)
    dt = dt.fit(data_train, res_train)
    return dt

def LR(data):
    data_train, data_test, res_train, res_test = data
    lr = LogisticRegression()
    res_train = res_train.values.ravel()
    lr = lr.fit(data_train, res_train)
    return lr

def RandomForest(data):
    data_train, data_test, res_train, res_test = data
    rf = RandomForestClassifier()
    res_train = res_train.values.ravel()
    rf = rf.fit(data_train, res_train)
    return rf

# подсчет отклонения
def balanced_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accuracy = balanced_accuracy_score(res_test, res_pred)
    return accuracy, res_pred

def f1_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accur = f1_score(res_test, res_pred, average="binary")
    return accur, res_pred

def fb_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accur = fbeta_score(res_test, res_pred, average="binary", beta=2)
    return accur, res_pred

# таблица полученных результатов
predict_data = (balanced_accuracy(KNN(data), X_test, Y_test)[1], balanced_accuracy(SVM(data), X_test, Y_test)[1], 
       balanced_accuracy(DecisionTree(data), X_test, Y_test)[1], balanced_accuracy(LR(data), X_test, Y_test)[1], 
       balanced_accuracy(RandomForest(data), X_test, Y_test)[1])
b_acc = [balanced_accuracy(KNN(data), X_test, Y_test)[0], balanced_accuracy(SVM(data), X_test, Y_test)[0], 
       balanced_accuracy(DecisionTree(data), X_test, Y_test)[0], balanced_accuracy(LR(data), X_test, Y_test)[0], 
       balanced_accuracy(RandomForest(data), X_test, Y_test)[0]]
f1_acc = [f1_accuracy(KNN(data), X_test, Y_test)[0], f1_accuracy(SVM(data), X_test, Y_test)[0], 
          f1_accuracy(DecisionTree(data), X_test, Y_test)[0], f1_accuracy(LR(data), X_test, Y_test)[0], 
          f1_accuracy(RandomForest(data), X_test, Y_test)[0], ]
fb_acc = [fb_accuracy(KNN(data), X_test, Y_test)[0], fb_accuracy(SVM(data), X_test, Y_test)[0], 
          fb_accuracy(DecisionTree(data), X_test, Y_test)[0], fb_accuracy(LR(data), X_test, Y_test)[0], 
          fb_accuracy(RandomForest(data), X_test, Y_test)[0], ]
fn = [confusion_matrix(Y_test, predict_data[0])[1][0], confusion_matrix(Y_test, predict_data[1])[1][0],
      confusion_matrix(Y_test, predict_data[2])[1][0], confusion_matrix(Y_test, predict_data[3])[1][0],
      confusion_matrix(Y_test, predict_data[4])[1][0]]
res = pd.DataFrame({
    "Models": ['KNN', 'Support vctors machine (SVM)', 'Decision Tree', 
               'Logistic Regression', 'Random Forest'],
    "balanced_accuracy": b_acc,
    "f1_accuracy": f1_acc,
    "fbeta_accuracy": fb_acc,
    "FN": fn,
})

# конфузионная матрица
title = f"Confusion matrix for {DecisionTree.__name__}, without normalization"

disp = ConfusionMatrixDisplay.from_estimator(
    DecisionTree(data),
    X_test,
    Y_test,
    cmap=plt.cm.Blues,
    normalize=None,
)
disp.ax_.set_title(title)

# Вывод
print('-'*60)
print(res)
print('-'*60)

# plt.show()

# print(f"До SMOTE")
# print(Y.value_counts())
# print("-"*50)
# print(f"После SMOTE")
# print(Y_res.value_counts())
