import numpy as np
import pandas as pd
import scipy.io as sio

def expression_fn(matrix,protein,alfabeto):
    res = np.zeros((matrix.shape[0],len(protein)))
    for i in range(len(protein)):
        index = [j for j,x in enumerate(alfabeto) if x == protein[i]]
        if index == []:
            res[:,i] = 0
        else:
            res[:,i] = matrix[:,index[0]]
    return res


def get_fea_mat():
    transfer_mat = [r'./data/all_whsx_list.mat',r'./data/energy_20.mat',r'./data/pssm.mat']
    tra_idx = ['all_whsx','energy_20','pssm']
    matrixs = []
    for i in range(len(transfer_mat)):
        matrix = sio.loadmat(transfer_mat[i])
        matrix = matrix[tra_idx[i]]
        matrixs.append(matrix)
    return matrixs


def fea_exp(protein):
    matrixs = get_fea_mat()
    alfabetos = []
    alfabetos.append(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
    alfabetos.append(['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'])
    features = []
    for i in range(len(matrixs)):
        if i == 0:
            res = expression_fn(matrixs[i], protein, alfabetos[0])
        else:
            res = expression_fn(matrixs[i], protein, alfabetos[1])
        df = pd.DataFrame(res)
        data = df.iloc[:,:].values
        data = data.reshape(-1,matrixs[i].shape[0])
        features.append(data)
    return features
