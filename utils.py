import os
import torch
import random
import numpy as np
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve


# Set a seed for training
def setup_seed(seed):
    np.random.seed(seed)                         # Numpy module.
    random.seed(seed)                            # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)                      # CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)             # GPU
        torch.cuda.manual_seed_all(seed)         # if you are using multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        

# load and process raw data
def load_data(file_dir):
    # output: 
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    drug_ids_dict, target_ids_dict = {},{}
    N, M, d_idx, t_idx = 0, 0, 0, 0  # N: the number of drug; M: the number of target
    data = []
    f = open(file_dir)
    for line in f.readlines():
        d, t = line.split()
        d = d.replace(':','')
        if d not in drug_ids_dict:
            drug_ids_dict[d] = d_idx
            d_idx += 1
        if t not in target_ids_dict:
            target_ids_dict[t] = t_idx
            t_idx += 1
        data.append([drug_ids_dict[d], target_ids_dict[t], 1])
    
    f.close()
    N, M = d_idx, t_idx

    return N, M, data, drug_ids_dict, target_ids_dict


# convert the list to a matrix
def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    mat = np.zeros([N,M])
    row = records_array[:,0].astype(np.int32)
    col = records_array[:,1].astype(np.int32)
    values = records_array[:,2].astype(np.float32)
    mat[row,col] = values
    
    return mat
    

# get AUC and AUPR scores    
def scores(y, y_):
    auc_score = roc_auc_score(y, y_)
    precision, recall, _ = precision_recall_curve(y, y_)
    aupr_score = auc(recall, precision)
    return auc_score, aupr_score


# get graph nodes by similarity 
def get_edge_index(mat, min_sim=0.1):
    N, M = mat.shape
    node0, node1 = [],[]
    for i in range(N):
        for j in range(M):
            if (mat[i,j] > min_sim) and (i != j):
                node0.append(i)
                node1.append(j)
    edge_index  = torch.tensor([node0, node1], dtype=torch.long)
    return edge_index