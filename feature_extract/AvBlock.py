import numpy as np

def GetAvBlock(P):
    l = P.shape[0]
    P = P.reshape(l,-1)
    # P:L*20
    blockLength = 3
    n = P.shape[0]
    B = int(np.floor(n/blockLength))
    AC = []
    for i in range(1,blockLength+1):
        RID = P[(B*(i-1))+1:B*i,:]
        mins = min(RID.shape)
        if mins == 1:
            AC.extend(RID)
        elif mins == 0:
            AC.extend(P[np.ceil(B*i),:])
        else:
            AC.extend(np.mean(RID,axis=0))
    return np.array(AC).reshape(1,-1)