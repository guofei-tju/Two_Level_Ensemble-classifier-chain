import numpy as np
import pywt
import cv2

def EstraggoConDWT(I,Nscale):
    F = []
    CA = I
    for i in range(Nscale):
        CA,CD = pywt.dwt(CA,'bior3.3')
        CA = CA.reshape(1,-1).astype(np.float32)
        CC = cv2.dct(CA)
        F.extend(CC[0,0:5])
        F.append(CA.min())
        F.append(CA.max())
        F.append(CA.mean())
        F.append(CA.std())
        F.append(CD.min())
        F.append(CD.max())
        F.append(CD.mean())
        F.append(CD.std())
    return F

def GetDWT(propertys):
    l = propertys.shape[0]
    propertys = propertys.reshape(-1,l)
    # property: 57*L
    n_propertys = propertys.shape[0]
    A = []
    for i in range(n_propertys):
        I_s = propertys[i,:]
        Fea = EstraggoConDWT(I_s,4)
        A.extend(Fea)
    return np.array(A).reshape(1,-1)