import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score,accuracy_score,make_scorer
from sklearn.preprocessing import StandardScaler,scale
from sklearn.metrics import roc_curve,auc
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings("ignore")


def load_data(filename):
    import pickle
    with open(filename,'rb') as f:
        model = pickle.load(f)
    return model

kfold = StratifiedKFold(n_splits=5)
def fit_model(classifier,X,y):
    scorer = make_scorer(accuracy_score)
    tuning = GridSearchCV(classifier,param_grid={},scoring=scorer,cv=kfold,n_jobs=2)
    tuning.fit(X,y)
    return tuning.best_estimator_

def select_model_train_with_vlaid(train_data,train_label,test_data,test_label):
    random_state = 0
    plt.figure(figsize=(12,8))
    plt.subplots_adjust(wspace=0.7, hspace=0.5)
    y = train_label.reshape(1,-1)[0]
    y_test = test_label.reshape(1,-1)[0]
    for i in range(len(train_data)):
        X = train_data[i]
        X_test = test_data[i]
        classifiers = []
        classifiers.append(LogisticRegression(random_state=random_state))
        classifiers.append(KNeighborsClassifier())
        classifiers.append(DecisionTreeClassifier(random_state=random_state))
        classifiers.append(SVC(probability=True,random_state=random_state))
        classifiers.append(RandomForestClassifier(random_state=random_state))
        # classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state))
        classifiers.append(GradientBoostingClassifier(random_state=random_state))
        classifiers.append(ExtraTreesClassifier(random_state=random_state))
        classifiers.append(lgb.LGBMClassifier(random_state=random_state))
        classifiers.append(xgb.XGBRFClassifier(random_state=random_state))
        acc = []
        for classifier in classifiers:
            clf = fit_model(classifier,X,y)
            y_pred = clf.predict(X_test)
            acc.append(accuracy_score(y_test,y_pred))
        indexs = ['Logistic','KNeighbors','DecisionTree','SVC','RandomForest','GradientBoosting','ExtraTreeClassifier','lightgbm','xgbRF']
        titles = ["PSE-PP","PSE-AAC","PSE-PSSM","AVB-PP","AVB-AAC","AVB-PSSM","DWT-PP","DWT-AAC","DWT-PSSM"]
        data = pd.DataFrame(acc,columns=['Acc'],index=indexs)
        print(titles[i])
        print(data)
        print()

        p = plt.subplot(3,3,(i+1))
        p.set_xlim([0,1])
        p.set_title(titles[i])
        # p.set_ylabel('Model')
        g = sns.barplot(x=data['Acc'],y=data.index,data=data)
    # plt.savefig("./imgs/model_select.jpg")
    plt.show()

def caucalate_metrics(y_true,y_pred):
    TP = np.sum(y_true[y_pred==1]==1)
    TN = np.sum(y_true[y_pred==0]==0)
    FN = np.sum(y_pred[y_true==1]==0)
    FP = np.sum(y_pred[y_true==0]==1)
    se = round(TP / (TP + FN),4)
    sp = round(TN / (FP + TN),4)
    acc = round((TP + TN) / len(y_true),4)
    a = (TP+FP)/10
    b = (TP+FN)/10
    c = (TN+FP)/10
    d = (TN+FN)/10
    mcc = round((((TP * TN) - (FP * FN)) / np.sqrt(a*b*c*d))/100,4)
    metrix = [se,sp,acc,mcc]
    return metrix

def cv(clf,X,y,kf):
    import matlab.engine
    import matlab
    eng = matlab.engine.start_matlab()

    metrics_total = []
    for train_idx, test_idx in kf.split(X, y):
        X_train,X_test = X[train_idx],X[test_idx]
        y_train,y_test = y[train_idx],y[test_idx]
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        # 多标签多分类_周志华代码
        y_test = matlab.double(y_test.T.tolist())
        y_pred = matlab.double(y_pred.T.tolist())
        acc = eng.Accuracy(y_pred, y_test)
        # 二分类指标
        # acc = accuracy_score(y_test,y_pred)
        metrics_total.append(acc)

        # metrics_total.append(caucalate_metrics(y_test,y_pred))
    return np.array(metrics_total)

def select_model(train_data,train_label):
    random_state = 0
    kfold = StratifiedKFold(n_splits=10)
    plt.figure(figsize=(12,8))
    plt.subplots_adjust(wspace=0.7, hspace=0.5)
    scorer = make_scorer(accuracy_score)
    for i in range(len(train_data)):
        X = scale(train_data[i])
        classifiers = []
        classifiers.append(LogisticRegression(random_state=random_state))
        classifiers.append(KNeighborsClassifier())
        classifiers.append(DecisionTreeClassifier(random_state=random_state))
        classifiers.append(SVC(probability=True,random_state=random_state))
        classifiers.append(RandomForestClassifier(random_state=random_state))
        classifiers.append(GradientBoostingClassifier(random_state=random_state))
        classifiers.append(ExtraTreesClassifier(n_estimators=120,bootstrap=True,random_state=random_state))
        cv_results = []
        for classifier in classifiers:
            # cv_result = cv(classifier, X, train_label,kfold)
            cv_result = cross_val_score(classifier,X,train_label,scoring=scorer,cv=kfold)
            cv_results.append(cv_result)
        cv_means = []
        cv_std = []
        for result in cv_results:
            cv_means.append(result.mean())
            cv_std.append(result.std())
        indexs = ['Logistic','KNeighbors','DecisionTree','SVC','RandomForest','GradientBoosting','ExtraTreeClassifier']
        titles = ["PSE-PP","PSE-AAC","PSE-PSSM","AVB-PP","AVB-AAC","AVB-PSSM","DWT-PP","DWT-AAC","DWT-PSSM"]
        data = pd.DataFrame(cv_means,columns=['Acc'],index=indexs)
        print(titles[i])
        print(data,"\n")

        p = plt.subplot(3,3,(i+1))
        p.set_xlim([0,1])
        p.set_title(titles[i])
        p.set_ylabel('Model')
        g = sns.barplot(x=data['Acc'],y=data.index,data=data)
    # plt.savefig("./imgs/model_select.jpg")
    plt.show()

def select_model_mutilabel(train_data,train_label):
    random_state = 0
    kf = KFold(n_splits=5)
    plt.figure(figsize=(12,8))
    plt.subplots_adjust(wspace=0.7, hspace=0.5)
    scorer = make_scorer(accuracy_score)
    for i in range(len(train_data)-1):
        X = scale(train_data[i])
        classifiers = []
        classifiers.append(OneVsRestClassifier(LogisticRegression(random_state=random_state)))
        classifiers.append(OneVsRestClassifier(KNeighborsClassifier()))
        classifiers.append(OneVsRestClassifier(DecisionTreeClassifier(random_state=random_state)))
        classifiers.append(OneVsRestClassifier(SVC(probability=True,random_state=random_state)))
        classifiers.append(OneVsRestClassifier(RandomForestClassifier(random_state=random_state)))
        classifiers.append(OneVsRestClassifier(GradientBoostingClassifier(random_state=random_state)))
        classifiers.append(OneVsRestClassifier(ExtraTreesClassifier(random_state=random_state)))
        cv_results = []
        for classifier in classifiers:
            cv_result = cv(classifier,X,train_label,kf)
            # cv_result = cross_val_score(classifier, X, train_label, scoring=scorer, cv=kf)
            cv_results.append(cv_result)
        cv_means = []
        for result in cv_results:
            cv_means.append(result.mean())
        indexs = ['Logistic','KNeighbors','DecisionTree','SVC','RandomForest','GradientBoosting','ExtraTreeClassifier']
        titles = ["PSE-PP","PSE-AAC","PSE-PSSM","AVB-PP","AVB-AAC","AVB-PSSM","DWT-PP","DWT-AAC","DWT-PSSM"]
        data = pd.DataFrame(cv_means,columns=['Acc'],index=indexs)
        print(titles[i])
        print(data,"\n")

        p = plt.subplot(3,3,(i+1))
        p.set_xlim([0,1])
        p.set_title(titles[i])
        # p.set_ylabel('Model')
        g = sns.barplot(x=data['Acc'],y=data.index,data=data)
    # plt.savefig("./imgs/model_select.jpg")
    plt.show()

def roc_auc(train_data,train_label,test_data,test_label):
    y = train_label.reshape(1, -1)[0]
    y_test = test_label.reshape(1, -1)[0]
    for i in range(len(train_data)):
        X = scale(train_data[i])
        X_test = scale(test_data[i])
        model = RandomForestClassifier()
        model.fit(X,y)
        y_score = model.predict_proba(X_test)
        fpr,tpr,_ = roc_curve(y_test,y_score)
        roc_auc = auc(fpr,tpr)
        print(roc_auc)

if __name__ == "__main__":

    # 加载第一层数据
    test_data = load_data("./data/ziti_train.pickle")

    for i in range(len(test_data)):
        print(test_data[i].shape)

    test_label = test_data[9].reshape(1, -1)[0]
    test_data.pop(9)
    select_model(test_data,test_label)

    # 加载第二层数据
    # test_data = load_data("./data/p2_ziti_ml_sample.pickle")
    # test_label = test_data[10]
    # test_data.pop(10)
    # for i in range(len(test_data)):
    #     model = ExtraTreesClassifier(n_estimators=120, random_state=0)
    #     test_data[i] = SelectFromModel(model,threshold="mean").fit_transform(test_data[i], test_label)
    #     print(test_data[i].shape)
    # print(test_label.shape)
    # select_model_mutilabel(test_data,test_label)

    # select_model_(train_data,train_label,test_data,test_label)