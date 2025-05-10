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
import shap

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

X_res, Y_res = randomoversample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2)
data = X_train, X_test, Y_train, Y_test
data_tr = X_train, Y_train

# используемые модели
def KNN(data):
    data_train, res_train = data
    model_neighbor = KNeighborsClassifier(n_neighbors=5, leaf_size=15, p=2)
    res_train = res_train.values.ravel()
    model_neighbor = model_neighbor.fit(data_train, res_train)
    return model_neighbor

def SVM(data):
    data_train, res_train = data
    clf = SVC(kernel='rbf', C=2.0, cache_size=200, degree=3, gamma=0.4, max_iter=200, tol=0.1)
    res_train = res_train.values.ravel()
    clf = clf.fit(data_train, res_train)
    return clf

def DecisionTree(data):
    data_train, res_train = data
    dt = DecisionTreeClassifier(random_state=0, max_depth=5, max_features=7, max_leaf_nodes=20, min_samples_leaf=1, min_samples_split=4)
    dt = dt.fit(data_train, res_train)
    return dt

def LR(data):
    data_train, res_train = data
    lr = LogisticRegression(C=5, intercept_scaling=1.0, max_iter=100)
    res_train = res_train.values.ravel()
    lr = lr.fit(data_train, res_train)
    return lr

def RandomForest(data):
    data_train, res_train = data
    rf = RandomForestClassifier(max_features=5, n_estimators=100, min_samples_leaf=1, min_samples_split=2)
    res_train = res_train.values.ravel()
    rf = rf.fit(data_train, res_train)
    return rf

# подсчет отклонения
def balanced_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accuracy = balanced_accuracy_score(res_test, res_pred)
    return accuracy

def f1_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accuracy = f1_score(res_test, res_pred, average="binary")
    return accuracy

def fb_accuracy(model, data_test, res_test):
    res_pred = model.predict(data_test)
    accuracy = fbeta_score(res_test, res_pred, average="binary", beta=2)
    return accuracy

knn_pred = KNN(data_tr).predict(X_test)
svm_pred = SVM(data_tr).predict(X_test)
dt_pred = DecisionTree(data_tr).predict(X_test)
lr_pred = LR(data_tr).predict(X_test)
rf_pred = RandomForest(data_tr).predict(X_test)


# таблица полученных результатов
predict_data = (knn_pred, svm_pred, dt_pred, lr_pred, rf_pred)
b_acc = [balanced_accuracy(KNN(data_tr), X_test, Y_test), balanced_accuracy(SVM(data_tr), X_test, Y_test), 
       balanced_accuracy(DecisionTree(data_tr), X_test, Y_test), balanced_accuracy(LR(data_tr), X_test, Y_test), 
       balanced_accuracy(RandomForest(data_tr), X_test, Y_test)]
f1_acc = [f1_accuracy(KNN(data_tr), X_test, Y_test), f1_accuracy(SVM(data_tr), X_test, Y_test), 
          f1_accuracy(DecisionTree(data_tr), X_test, Y_test), f1_accuracy(LR(data_tr), X_test, Y_test), 
          f1_accuracy(RandomForest(data_tr), X_test, Y_test), ]
fb_acc = [fb_accuracy(KNN(data_tr), X_test, Y_test), fb_accuracy(SVM(data_tr), X_test, Y_test), 
          fb_accuracy(DecisionTree(data_tr), X_test, Y_test), fb_accuracy(LR(data_tr), X_test, Y_test), 
          fb_accuracy(RandomForest(data_tr), X_test, Y_test), ]
fn = [confusion_matrix(Y_test, predict_data[0])[1][0], confusion_matrix(Y_test, predict_data[1])[1][0],
      confusion_matrix(Y_test, predict_data[2])[1][0], confusion_matrix(Y_test, predict_data[3])[1][0],
      confusion_matrix(Y_test, predict_data[4])[1][0]]
res = pd.DataFrame({
    "Models": ['KNN', 'Support vctors machine (SVM)', 'Decision Tree', 
               'Logistic Regression', 'Random Forest'],
    "balanced_accuracy": b_acc,
    "f1_accuracy": f1_acc,
    "f2_accuracy": fb_acc,
    "FN": fn,
})

# конфузионная матрица
title = f"Confusion matrix for {DecisionTree.__name__}, without normalization"

disp = ConfusionMatrixDisplay.from_estimator(
    DecisionTree(data_tr),
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

# значения Шепли
def Forestshap(datatr):
    plt.clf()
    expl_rf = shap.Explainer(RandomForest(datatr))
    shap_values = expl_rf(datatr[0])
    shap_values.values = shap_values.values[:,:,0]
    shap_values.base_values = shap_values.base_values[0,1]
    # shap.plots.waterfall(shap_values[1])
    # shap.summary_plot(shap_values)
    # shap.plots.beeswarm(shap_values)
    shap.plots.bar(shap_values[0], show=False)
    plt.title("Shap value for Random Forest(random oversampling)")
    plt.show()
    

def DecTreeshap(datatr):
    plt.clf()
    exp_rf = shap.Explainer(DecisionTree(datatr))
    shap_values = exp_rf(datatr[0])
    shap_values.values = shap_values.values[:,:,0]
    shap_values.base_values = shap_values.base_values[0,1]
    shap.plots.bar(shap_values, show=False)
    plt.title("Shap value for Decision Tree(random oversampling)")
    plt.show()
    
def LinRegrshap(datatr):
    plt.clf()
    exp_rf = shap.Explainer(LR(datatr), X_train)
    shap_values = exp_rf(datatr[0])
    # shap_values.values = shap_values.values[:,:,0]
    # shap_values.base_values = shap_values.base_values[0,1]
    shap.plots.bar(shap_values, show=False)
    plt.title("Shap value for Logistic Regression(random oversampling)")
    plt.show()

Forestshap(data_tr)
DecTreeshap(data_tr)
LinRegrshap(data_tr)

# plt.show()

# проверка smote и randomoversampling
# print(f"До SMOTE")
# print(Y.value_counts())
# print("-"*50)
# print(f"После SMOTE")
# print(Y_res.value_counts())
