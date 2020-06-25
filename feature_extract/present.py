import numpy as np
import test_for_paper2.extra as ext
import test_for_paper2.express as exp


def read_AMP(dirname,filename):
    sequences = []
    lines = ""
    with open("./"+dirname+"/"+filename+".fasta",'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('>'):
                if lines != "" and len(lines)>5:
                    sequences.append(lines)
                lines = ""
                continue
            if line == '\n':
                continue
            line = line.split()
            lines += line[0]
    return sequences


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
    all_data = []
    for i in range(len(names)):
        data = read_CDHIT_seq(dirname, names[i])
        # print(names[i],len(data))
        all_data.extend(data)
    # print("全部肽：", len(all_data))
    all_data = list(set(all_data))
    # print("去重后：", len(all_data))
    return all_data


def read_neg(dirname,filename):
    mp = {"antibacterial": 1, "antibiofilm": 1, "anti-MRSA": 1, "antiviral": 1, "anti-HIV": 1, "antifungal": 1,"anti-protist": 1,"antiparasital": 1, "antimalarial": 1, "anticancer": 1,
          "spermicidal": 1, "insecticidal": 1,"chemotactic ": 1, "anti-inflamm": 1,"wound": 1, "channel": 1, "anti-toxin": 1, "protease": 1, "antioxidant": 1, "antiparasital": 1,
          "Uncharacterized": 1, "Metalloprotease": 1,"Zinc": 1, "Apolipoprotein": 1, "Transferrin": 1, "DNA-directed": 1, "Splicing": 1, "Urea": 1,"Keratin-associated": 1,
          "Putative": 1,"NADH": 1, "Histone": 1, "Choriogonadotropin": 1, "Pancreas/duodenum": 1, "GTP-binding": 1, "Signal": 1,"Kelch-like": 1, "Putative": 1,
          "Transcriptional": 1, "Transmembrane": 1, "Solute": 1, "Rhophilin-2": 1, "TBC1": 1, "Synaptopodin": 1,"Tetraspanin-32": 1, "Signal": 1}
    sequences = []
    lines = ""
    with open("./"+dirname+"/"+filename+".fasta",'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('>'):
                if lines != "" and len(lines)<130:
                    line = line.split()
                    if mp.get(line[1]) == None:
                        # sequences.append(line[0]+" "+line[1])
                        print(lines)
                        sequences.append(lines)
                lines = ""
                continue
            line = line.split()
            lines += line[0]
    return np.array(sequences)


def load_neg():
    neg = read_neg("data", 'neg')
    # print("负样本数目: ", len(neg))
    np.random.seed(7)
    del_idx = set(np.random.randint(0, 766, 56))
    # print("删除随机生成的 " + str(len(del_idx)) + " 个索引")
    all = set(np.arange(0, 766))
    all = all - del_idx
    neg = neg[list(all)]
    # print("删除后负样本个数: ", len(neg))
    return neg


def save_tmp(data,filename):
    import pickle
    with open(filename,'wb') as f:
        pickle.dump(data,f)


def read_paper1(filename):
    sequences = []
    with open("./data/" + filename + ".txt", 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()
            sequences.append(line[-1][2:])
    return sequences


def write_data(filename,data):
    fh = open(filename,'w')
    for i in range(len(data)):
        fh.write(data[i])
        fh.write("\n")
    fh.close()

def Get_data():
    # ziti数据
    # pos_seqs = load_pos("data")
    # neg_seqs = list(load_neg())

    # MLAMP数据
    p2_pos_seqs = read_AMP("data", "p2_validation_amp")
    # p2_pos_seqs.extend(read_AMP("data", "p2_validation_amp"))
    # p2_neg_seqs = read_AMP("data", "p2_validation_nonamp")
    # p2_neg_seqs.extend(read_AMP("data", "p2_validation_nonamp"))

    # ziti_pos_seqs = load_pos("data")
    # ziti_neg_seqs = list(load_neg())

    print("p2_pos:", len(p2_pos_seqs))
    # print("p2_neg:", len(p2_neg_seqs))
    # print("ziti_pos:", len(ziti_pos_seqs))
    # print("ziti_neg:", len(ziti_neg_seqs))

    # pos_seqs = list(set(p2_pos_seqs) - set(ziti_pos_seqs))
    # neg_seqs = list(set(p2_neg_seqs) - set(ziti_neg_seqs))
    # print(len(pos_seqs))
    # print(len(neg_seqs))

    # pos = len(pos_seqs)
    # neg = len(neg_seqs)
    # print("正样本序列：",pos)
    # print("负样本序列：",neg)
    #
    # pos_seqs.extend(neg_seqs)
    # print(len(pos_seqs))
    dictMat = {}
    for i in range(len(p2_pos_seqs)):
        # s[0]:l*57理化性质
        # s[1]:l*20相互作用
        # s[2]:pssm矩阵
        s = exp.fea_exp(p2_pos_seqs[i])
        if i == 0:
            dictMat[0] = ext.extra_pse(s[0])
            dictMat[1] = ext.extra_pse(s[1])
            dictMat[2] = ext.extra_pse(s[2])

            dictMat[3] = ext.extra_avb(s[0])
            dictMat[4] = ext.extra_avb(s[1])
            dictMat[5] = ext.extra_avb(s[2])

            dictMat[6] = ext.extra_DWT(s[0])
            dictMat[7] = ext.extra_DWT(s[1])
            dictMat[8] = ext.extra_DWT(s[2])
        else:
            dictMat[0] = np.vstack([dictMat[0],ext.extra_pse(s[0])])
            dictMat[1] = np.vstack([dictMat[1],ext.extra_pse(s[1])])
            dictMat[2] = np.vstack([dictMat[2],ext.extra_pse(s[2])])

            dictMat[3] = np.vstack([dictMat[3],ext.extra_avb(s[0])])
            dictMat[4] = np.vstack([dictMat[4],ext.extra_avb(s[1])])
            dictMat[5] = np.vstack([dictMat[5],ext.extra_avb(s[2])])

            dictMat[6] = np.vstack([dictMat[6],ext.extra_DWT(s[0])])
            dictMat[7] = np.vstack([dictMat[7],ext.extra_DWT(s[1])])
            dictMat[8] = np.vstack([dictMat[8],ext.extra_DWT(s[2])])

    # 标签
    # dictMat[9] = np.hstack([np.ones((1,pos)),np.zeros((1,neg))])
    dictMat[9] = np.ones((1,len(p2_pos_seqs)))
    print(dictMat[0].shape)
    print(dictMat[1].shape)
    print(dictMat[2].shape)

    save_tmp(dictMat,"./data/mlamp_validation_920.pickle")


def p2_expect_ziti():
    p2_pos_seqs = read_AMP("data","p2_validation_amp")
    # p2_pos_seqs.extend(read_AMP("data","p2_validation_amp"))
    p2_neg_seqs = read_AMP("data","p2_validation_nonamp")
    # p2_neg_seqs.extend(read_AMP("data","p2_validation_nonamp"))

    ziti_pos_seqs = load_pos("data")
    ziti_neg_seqs = list(load_neg())

    print("p2_pos:",len(p2_pos_seqs))
    print("p2_neg:",len(p2_neg_seqs))
    print("ziti_pos:",len(ziti_pos_seqs))
    print("ziti_neg:",len(ziti_neg_seqs))

    pos_seqs = list(set(p2_pos_seqs)-set(ziti_pos_seqs))
    neg_seqs = list(set(p2_neg_seqs)-set(ziti_neg_seqs))
    print(len(pos_seqs))
    print(len(neg_seqs))


if __name__ == "__main__":

    Get_data()
    # p2_expect_ziti()

    # neg_seqs = list(load_neg())
    # print(neg_seqs)