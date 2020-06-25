import numpy as np

AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
def GetASDC(seq):
    Mat = np.zeros(400)
    sum = 0
    index = 0
    for i in range(20):
        for j in range(20):
            X = AA[i]
            Y = AA[j]
            cnt_pair = 0
            m = 0
            l = len(seq)
            while m < l:
                n = m + 1
                if seq[m] == X:
                    while n < l and n != m:
                        if seq[n] == Y:
                            cnt_pair = cnt_pair + 1
                        n = n + 1
                m = m + 1
            sum = sum + cnt_pair
            Mat[index] = cnt_pair
            index = index + 1
    for p in range(400):
        Mat[p] = Mat[p] / sum
    return np.array(Mat).reshape(1,-1)