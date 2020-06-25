import numpy as np

def GetPse(P,lg):
    l = P.shape[0]
    P = P.reshape(l, -1)
    # P:L*20
    fea = []
    n = P.shape[0]
    V = P
    AC = np.zeros((20,lg))
    for lag in range(lg):
        for i in range(20):
            for j in range(n-lag):
                AC[i,lag] = AC[i,lag] + (V[j,i]-V[j+lag,i])**2
            if n-lag != 0:
                AC[i,lag] = AC[i,lag]/(n-lag)
    fea.extend(np.mean(V,axis=0))
    AC = AC.reshape(1,-1)
    fea.extend(AC[0])
    return np.array(fea).reshape(1,-1)

# arr = np.zeros((9,20))
# print(arr.shape)
# ans = GetPse(arr,3)
# print(ans.shape)