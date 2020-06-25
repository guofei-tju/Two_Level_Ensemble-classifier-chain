import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold,KFold
from imblearn.over_sampling import SMOTE,SVMSMOTE,KMeansSMOTE,ADASYN
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.multioutput import ClassifierChain

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
import test_for_paper2.extra as ext
import test_for_paper2.express as exp
import test_for_paper2.wangye as wy

import warnings
warnings.filterwarnings('ignore')

def save_tmp(data,filename):
    import pickle
    with open(filename,'wb') as f:
        pickle.dump(data,f)

def read_CDHIT_seq(dirname,filename):
    sequences = []
    with open("./"+dirname+"/"+filename+".txt",'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line == '\n' or line.startswith('>'):
                continue
            line = line.split()
            if len(line[0]) < 130:
                sequences.append(line[0])
    return sequences

def load_pos(dirname):
    names = ['antibacterial','anticancer','antifungal','anti-HIV','anti-MRSA','antiparasital','antiviral']
    mp = {}
    X,y = [],[]
    for i in range(len(names)):
        data = read_CDHIT_seq(dirname, names[i])
        for j in range(len(data)):
            if data[j] in mp.keys():
                mp[data[j]].append(i)
            else:
                mp[data[j]] = [i]
        print(names[i],len(data))
    for key,val in mp.items():
        X.append(key)
        y.append(val)
    return X,y

def Get_data():
    X,y = load_pos("data")
    print(X)
    dictMat = {}
    for i in range(len(X)):
        s = exp.fea_exp(X[i])
        if i == 0:
            # dictMat[0] = ext.extra_pse(s[0])
            # dictMat[1] = ext.extra_pse(s[1])
            # dictMat[2] = ext.extra_pse(s[2])
            #
            # dictMat[3] = ext.extra_avb(s[0])
            # dictMat[4] = ext.extra_avb(s[1])
            # dictMat[5] = ext.extra_avb(s[2])
            #
            # dictMat[6] = ext.extra_DWT(s[0])
            # dictMat[7] = ext.extra_DWT(s[1])
            # dictMat[8] = ext.extra_DWT(s[2])

            dictMat[9] = ext.extra_asdc(X[i])
        else:
            # dictMat[0] = np.vstack([dictMat[0],ext.extra_pse(s[0])])
            # dictMat[1] = np.vstack([dictMat[1],ext.extra_pse(s[1])])
            # dictMat[2] = np.vstack([dictMat[2],ext.extra_pse(s[2])])
            #
            # dictMat[3] = np.vstack([dictMat[3],ext.extra_avb(s[0])])
            # dictMat[4] = np.vstack([dictMat[4],ext.extra_avb(s[1])])
            # dictMat[5] = np.vstack([dictMat[5],ext.extra_avb(s[2])])
            #
            # dictMat[6] = np.vstack([dictMat[6],ext.extra_DWT(s[0])])
            # dictMat[7] = np.vstack([dictMat[7],ext.extra_DWT(s[1])])
            # dictMat[8] = np.vstack([dictMat[8],ext.extra_DWT(s[2])])

            dictMat[9] = np.vstack([dictMat[9],ext.extra_asdc(X[i])])

    # dictMat[9] = y
    # save_tmp(dictMat, "./data/p2_ziti_ml_asdc.pickle")

def load_data(filename):
    import pickle
    with open(filename,'rb') as f:
        model = pickle.load(f)
    return model

def print_metrics(y_test,y_pred,y_score):
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    precision = round(precision_score(y_test, y_pred, average="samples"),4)
    recall = round(recall_score(y_test, y_pred, average="samples"),4)

    import matlab.engine
    import matlab
    eng = matlab.engine.start_matlab()
    y_test = matlab.double(y_test.T.tolist())
    y_pred = matlab.double(y_pred.T.tolist())
    acc = eng.Accuracy(y_pred,y_test)
    ap = eng.Average_precision(y_pred,y_test)
    cov = eng.coverage(y_pred,y_test)
    hl = eng.Hamming_loss(y_pred,y_test)
    oe = eng.One_error(y_pred,y_test)
    rl = eng.Ranking_loss(y_pred,y_test)
    return [acc,ap,cov,hl,oe,rl,precision,recall]


def multilabel_loo(data,y,lian):
    kf = KFold(n_splits=10)
    chain = OneVsRestClassifier(ExtraTreesClassifier(bootstrap=True,n_estimators=120),n_jobs=8)
    chains = [ClassifierChain(chain,order="random") for i in range(lian)]
    model = OneVsRestClassifier(ExtraTreesClassifier(bootstrap=True,n_estimators=200), n_jobs=8)
    metrics_total = []
    fea_train = np.array([])
    fea_test = np.array([])
    # for train_idx, test_idx in loo.split(data[0]):
    for train_idx, test_idx in kf.split(data[0],y):
        y_train, y_test = y[train_idx], y[test_idx]
        for i in range(lian):
            X_train,X_test = data[i%10][train_idx],data[i%10][test_idx]
            clf = chains[i]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if i == 0:
                fea_train = clf.predict(X_train)
                fea_test = y_pred
            else:
                fea_train = np.hstack([fea_train,clf.predict(X_train)])
                fea_test = np.hstack([fea_test,y_pred])
        print(fea_train.shape)
        fea_train,fea_test = wy.fea_extra(fea_train,y_train,fea_test,y_test)
        print(fea_train.shape)

        model.fit(fea_train,y_train)
        y_pred = model.predict(fea_test)
        y_score = model.predict_proba(fea_test)
        metrics = print_metrics(y_test, y_pred, y_score)
        print(y_test.shape,y_pred[:3])
        metrics_total.append(metrics)
    print(["Accuracy","Average_precision","coverage","Hamming_loss","One_error","Ranking_loss","Precision","Recall"])
    print(lian,'\n',metrics_total)
    print(np.mean(metrics_total, axis=0))


def train_and_pred(dictTrainMats,Trainlabel,dictTestMats,lian):
    chain = OneVsRestClassifier(ExtraTreesClassifier(bootstrap=True, n_estimators=120), n_jobs=8)
    chains = [ClassifierChain(chain, order="random") for i in range(lian)]
    model = OneVsRestClassifier(ExtraTreesClassifier(bootstrap=True, n_estimators=200), n_jobs=8)
    fea_train = np.array([])
    fea_test = np.array([])
    for i in range(lian):
        X_train, X_test = dictTrainMats[i%8],dictTestMats[i%8]
        clf = chains[i]
        clf.fit(X_train, Trainlabel)
        y_pred = clf.predict(X_test)
        if i == 0:
            fea_train = clf.predict(X_train)
            fea_test = y_pred
        else:
            fea_train = np.hstack([fea_train, clf.predict(X_train)])
            fea_test = np.hstack([fea_test, y_pred])
    print(fea_train.shape,fea_test.shape)
    model.fit(fea_train, Trainlabel)
    y_pred = model.predict(fea_test)
    print(y_pred.shape)
    save_tmp(y_pred, "./data/mlamp_train_710Test.pickle")


def fit_model(classifier,params,X,y,kfold):
    scorer = make_scorer(accuracy_score)
    tuning = GridSearchCV(classifier,param_grid=params,scoring=scorer,cv=kfold,n_jobs=-1)
    tuning.fit(X,y)
    print(tuning.best_params_)
    print(tuning.best_score_)
    return tuning.best_estimator_

def down_sample(y):
    idxs = []
    antibacterial, antifungal = [], []
    for i in range(len(y)):
        if y[i] == [0]:
            antibacterial.append(i)
        elif y[i] == [2]:
            antifungal.append(i)
        else:
            idxs.append(i)
    np.random.seed(0)
    antibacterial = np.array(antibacterial)
    antifungal = np.array(antifungal)
    random_sel = list(set(np.random.randint(0, 96, 40)))
    idxs.extend(antibacterial[random_sel])
    idxs.extend(antifungal[random_sel])
    return idxs

def up_sample(X,y,num):
    X_res,y_res = [],[]
    for i in range(7):
        idxs = []
        for idx in range(len(y)):
            if i in y[idx]:
                idxs.append(idx)
        X_res.extend(X[idxs])
        y_res.extend(len(idxs)*[i])
    X_res = np.array(X_res)
    y_res = np.array(y_res)
    print(y_res)
    print(Counter(y_res))
    # clf = SVC(kernel="rbf", C=0.01, gamma="scale",decision_function_shape="ovo", probability=True)
    # sm = SVMSMOTE(k_neighbors=1,m_neighbors=5,svm_estimator=clf)
    # sm = KMeansSMOTE(kmeans_estimator=KMeans(n_clusters=8,n_init=20))
    sm = ADASYN()
    X_resample,y_resample = sm.fit_sample(X_res,y_res)
    print(Counter(y_resample))
    tmp = []
    X = np.vstack([X,X_resample[y_resample==6][:num]])
    for i in range(num):
        tmp.append([6])
    X = np.vstack([X, X_resample[y_resample==4][:num]])
    for i in range(num):
        tmp.append([4])
    X = np.vstack([X, X_resample[y_resample==1][:num]])
    for i in range(num):
        tmp.append([1])
    X = np.vstack([X, X_resample[y_resample==3][:num]])
    for i in range(num):
        tmp.append([3])
    X = np.vstack([X, X_resample[y_resample==5][:num]])
    for i in range(num):
        tmp.append([5])
    t = y.tolist()
    t.extend(tmp)
    y = np.array(t)
    print(X.shape)
    print(y.shape)
    return X,y

def sample(num):
    # 加载未采样数据数据
    data = load_data("./data/p2_ziti_ml.pickle")
    y = np.array(data[9])
    # y = MultiLabelBinarizer().fit_transform(y)
    data.pop(9)
    asdc = load_data("./data/p2_ziti_ml_asdc.pickle")
    data[9] = asdc[9]
    # 采样
    for i in range(len(data)):
        data[i],y_ = up_sample(data[i],y,num)
    # 多标签多分类编码
    y = MultiLabelBinarizer().fit_transform(y_)
    data[10] = y
    # save_tmp(data, "./data/ziti_ml.pickle")

if __name__ == "__main__":

    # multilLabel()
    # Get_data()

    # 采样
    sample(num=0)

    # 加载采样后的数据
    # data = load_data("./data/p2_ziti_ml_sample_adasyn.pickle")
    # y = data[10]
    # data.pop(10)
    # for i in range(len(data)):
    #     data[i] = SelectFromModel(ExtraTreesClassifier(n_estimators=120,random_state=0), threshold="mean").fit_transform(data[i], y)
    #     print(data[i].shape)
    # print(y.shape)

    # 特征预处理
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    # 网格调参
    # clf = OneVsRestClassifier(ExtraTreesClassifier(bootstrap=True,n_estimators=120))
    # print(clf.get_params())
    # param = {"estimator__n_estimators":[100,120,150,200]}
    # fit_model(clf,param,X,y,5)

    # 十折交叉验证
    # for i in np.linspace(10,200,20,dtype=np.int):
    #     multilabel_loo(data,y,i)

