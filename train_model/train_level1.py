import numpy as np
import test_for_paper2.wangye as wy
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve,auc
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data(filename):
    import pickle
    with open(filename,'rb') as f:
        model = pickle.load(f)
    return model

def save_model(model,filename):
    import pickle
    with open(filename,'wb') as f:
        pickle.dump(model,f)

def load_model(filename):
    import pickle
    with open(filename,'rb') as f:
        model = pickle.load(f)
    return model


def fit_model(classifier,params,X,y,kfold):
    scorer = make_scorer(accuracy_score)
    tuning = GridSearchCV(classifier,param_grid=params,scoring=scorer,cv=kfold,n_jobs=4)
    tuning.fit(X,y)
    print(tuning.best_params_)
    print(tuning.best_score_)
    return tuning.best_estimator_

def train(X,y):
    y = y.reshape(1,-1)[0]
    # clf_rf = RandomForestClassifier(random_state=1)
    # param_rf = {"n_estimators": np.arange(100, 151, 10)}
    # clf_rf_best = fit_model(clf_rf, param_rf, X, y)
    # labelPre = clf_rf_best.predict(X)
    # clf_best = ExtraTreesClassifier(bootstrap=True,n_estimators=120)
    clf_best = GradientBoostingClassifier(n_estimators=80)
    clf_best.fit(X,y)
    labelPre = clf_best.predict(X)
    return clf_best,labelPre[:,np.newaxis]

def randSelect(dataIndex,i):
    j = dataIndex.index(i)
    while i == dataIndex[j]:
            j = int(np.random.uniform(0,len(dataIndex)))
    return j

def oneChain(dictMats,label,last):
    l = len(dictMats)
    dataIndex = list(range(l))
    chains = []
    save_index = []
    labelPres = np.array([])
    for i in range(l-1):
        randomIndex = randSelect(dataIndex,last)
        index = dataIndex[randomIndex]
        if i == 0:
            model,labelPre = train(dictMats[index],label)
            labelPres = labelPre
        else:
            model,labelPre = train(np.hstack([dictMats[index],labelPres]),label)
            labelPres = np.hstack([labelPres,labelPre])
        chains.append(model)
        del (dataIndex[randomIndex])
        save_index.append(str(index))
    model, labelPre = train(np.hstack([dictMats[last], labelPres]),label)
    chains.append(model)
    save_index.append(str(last))
    # print('randIndex_list :',save_index)
    labelPres = np.hstack([labelPres, labelPre])
    return chains,save_index,labelPre

def multiCC(dictMats,label,lian):
    multiClissiferChain = []
    multiSaveIndex = []
    mod = len(dictMats)
    for i in range(lian):
        # print("Chain -----------------------------------------> ",i)
        chain,save_index,labelOfOneChian = oneChain(dictMats,label,i%mod)
        multiClissiferChain.append(chain)
        multiSaveIndex.append(save_index)
        if i == 0:
            pEncoder = labelOfOneChian
        else:
            pEncoder = np.hstack([pEncoder,labelOfOneChian])
    return pEncoder,multiClissiferChain,multiSaveIndex


def train_model(dictMats,label):
    X,ccModel,ccIndex = multiCC(dictMats,label)
    print(X.shape)

    rf = RandomForestClassifier(random_state=1)
    param_rf = {"min_samples_split":[2,3],
                "n_estimators": [50,60,70,80,100,120,150,160,200,220,300,350]}

    gbdt = GradientBoostingClassifier(random_state=1)
    param_gbdt = {"n_estimators": [10, 20, 50, 100, 300, 500]}

    label = label.reshape(1, -1)[0]
    model = fit_model(rf,param_rf,X,label)

    save_model(model, r'./model/com_train_rf_clf.pickle')
    save_model(ccModel, r'./model/com_train_rf_CCmodel.pickle')
    save_model(ccIndex, r'./model/com_train_rf_index.pickle')
    print("Train Finish!")


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

    # fpr, tpr, _ = roc_curve(y_true, y_ps[:, 1])
    # roc_auc = auc(fpr, tpr)
    metrix = [se, sp, acc, mcc]
    return metrix


def encoder(dictMats,ccModel,ccIndex,lian):
    labelPres = np.array([])
    for i in range(lian):
        for j in range(len(dictMats)):
            clf = ccModel[i][j]
            index = ccIndex[i][j]
            if j == 0:
                labelPre = clf.predict(dictMats[int(index)])
                labelPres = labelPre[:,np.newaxis]
            else:
                # labelPre = clf.predict(dictMats[int(index)])
                labelPre = clf.predict(np.hstack([dictMats[int(index)],labelPres]))
                labelPres = np.hstack([labelPres,labelPre[:,np.newaxis]])
        labelPre = labelPre.reshape(-1,1)
        if i == 0:
            pEncoder = labelPre
        else:
            pEncoder = np.hstack([pEncoder,labelPre])
    return pEncoder


def make_predict(dictTestMats,Testlabel,lian):
    ccfile = r'./model/zitiTrain_920Test_CCmodel.pickle'
    clffile = r'./model/zitiTrain_920Test_clf.pickle'
    indexfile = r'./model/zitiTrain_920Test_index.pickle'
    ccModel = load_model(ccfile)
    ccIndex = load_model(indexfile)
    clf = load_model(clffile)
    X_test = encoder(dictTestMats,ccModel, ccIndex,lian)
    y = Testlabel.reshape(1,-1)[0]
    y_pred = clf.predict(X_test)

    # 画 roc 曲线
    y_ps = clf.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y, y_ps[:,1])
    roc_auc = auc(fpr, tpr)
    print("AUC :",roc_auc)
    plt.plot(fpr,tpr)
    print("score: ", caucalate_metrics(y, y_pred))
    plt.show()


def cross_val(dictMats,label,lian):
    label = label.reshape(1, -1)[0]
    dictTrainM = {}
    dictTestM = {}
    skf = StratifiedKFold(n_splits=10)
    # kf = KFold(n_splits=10)
    scores = []
    rf = RandomForestClassifier(n_estimators=200)
    param_rf = {"min_samples_split": [2, 3],
                "n_estimators": [50, 60, 70, 80, 100, 120, 150, 160, 200]}
    for train_index, test_index in skf.split(dictMats[0],label):
        for i in range(len(dictMats)):
            dictTrainM[i] = dictMats[i][train_index]
            dictTestM[i] = dictMats[i][test_index]
        train_label = label[train_index]
        test_label = label[test_index]
        X_train,ccModel,ccIndex = multiCC(dictTrainM,train_label,lian)
        X_test = encoder(dictTestM,ccModel, ccIndex,lian)
        print(X_test.shape)
        fea_train, fea_test = wy.fea_extra(X_train, train_label.reshape(-1,1), X_test, test_label.reshape(-1,1))
        # model = fit_model(rf, param_rf, X_train, train_label,5)
        rf.fit(fea_train,train_label)

        y_pred = rf.predict(fea_test)
        y_ps = rf.predict_proba(fea_test)
        scores.append(caucalate_metrics(test_label,y_pred,y_ps))
    print(lian,'\n',scores)
    print(np.mean(scores,axis=0))
    return scores


def train_and_pred(dictTrainMats,Trainlabel,dictTestMats,Testlabel,lian,epoch):
    acc = 0.00
    label = Trainlabel.reshape(1, -1)[0]
    y = Testlabel.reshape(1,-1)[0]
    for e in range(epoch):
        X_train, ccModel, ccIndex = multiCC(dictTrainMats,Trainlabel,lian)
        X_test = encoder(dictTestMats, ccModel, ccIndex,lian)

        tr_d,te_d = wy.fea_extra(X_train, label, X_test, y)
        print(te_d.shape)

        model = RandomForestClassifier(n_estimators=150)
        model.fit(tr_d,label)

        y_pred = model.predict(te_d)
        score = caucalate_metrics(y, y_pred)
        if score[2] > acc:
            print("Epoch: {}/{}, acc increase from {} to {}".format(e,epoch,acc,score[2]))
            print(score)
            acc = score[2]
            # save_model(ccModel, r'./model/zitiTrain_920Test_CCmodel.pickle')
            # save_model(ccIndex, r'./model/zitiTrain_920Test_index.pickle')
            # save_model(model, r'./model/zitiTrain_920Test_clf.pickle')
        else:
            print("Epoch: {}/{}".format(e,epoch))


if __name__ == "__main__":

    train_data = load_data("./data/p2_com_train.pickle")
    test_data = load_data("./data/ziti_train.pickle")

    for i in range(len(train_data)):
        print(train_data[i].shape)
    for i in range(len(test_data)):
        print(test_data[i].shape)

    train_label = train_data[9].reshape(1,-1)
    train_data.pop(9)
    test_label = test_data[9].reshape(1,-1)
    test_data.pop(9)

    # train_model(train_data,train_label)
    # for i in np.linspace(10,200,20,dtype=np.int):
    #     cross_val(test_data,test_label,i)

    # make_predict(test_data,test_label,90)

    for i in np.linspace(10,200,20,dtype=np.int):
        train_and_pred(train_data,train_label,test_data,test_label,i,10)
