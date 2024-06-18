import numpy as np
import pandas as pd
import random
from torch_geometric.data import Data
import matplotlib.pyplot as plt
# from scipy import interp
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Any
import torch
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x


def constructNet(association_matrix): # construct association matrix
    n, m = association_matrix.shape
    drug_matrix = torch.zeros((n, n), dtype=torch.int8)
    meta_matrix = torch.zeros((m, m), dtype=torch.int8)
    mat1 = torch.cat((drug_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), meta_matrix), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0


def load_data(seed, n_components):
    Adj = pd.read_csv('../data1/association_matrix.csv', header=0)
    count_ones = np.count_nonzero(Adj == 1)
    print("元素为1的个数：", count_ones)

    Dis_simi = pd.read_csv('../data1/diease_simi_network.csv', header=0)
    Dis_adj = np.where(Dis_simi > 0.4, 1, 0)
    count_ones_disease = np.count_nonzero(Dis_adj)
    print("Dis_adj 中值为 1 的元素个数:", count_ones_disease)
    Dis_adj = torch.tensor(Dis_adj).to(device)

    Meta_simi = pd.read_csv('../data1/metabolite_simi_ntework.csv', header=0)
    Meta_adj = np.where(Meta_simi > 0.4, 1, 0)
    count_ones_meta = np.count_nonzero(Meta_adj)
    print("Meta_adj 中值为 1 的元素个数:", count_ones_meta)
    Meta_adj = torch.tensor(Meta_adj).to(device)

    Dis_MESH2vec = pd.read_csv('../data1/MeSHHeading2vec.csv', header=0)
    Meta_mol2vec = pd.read_csv('../data1/metabolite_mol2vec.csv', header=0)

    # PCA
    pca = PCA(n_components=n_components)
    PCA_dis_feature = pca.fit_transform(Dis_MESH2vec.values)
    PCA_metabolite_feature = pca.fit_transform(Meta_mol2vec.values)
    Dis_feature = torch.FloatTensor(PCA_dis_feature).to(device)
    Meta_feature = torch.FloatTensor(PCA_metabolite_feature).to(device)
    feature = torch.cat((Dis_feature, Meta_feature), dim=0).to(device)

    # Training and validation set samples
    index_matrix = np.mat(np.where(Adj == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    return Adj, Dis_adj, Meta_adj, feature, random_index, k_folds


def laplacian_positional_encoding(adj, pe_dim):
    N = torch.diag(torch.pow(torch.sum(adj, dim=1).clamp(min=1), -0.5))  # 归一化矩阵N
    L = torch.eye(adj.shape[0]).to(device) - N @ adj @ N
    EigVal, EigVec = torch.linalg.eig(L)
    EigVal = EigVal.real
    EigVec = EigVec.real
    sorted_indices = EigVal.argsort()
    EigVec_sorted = EigVec[:, sorted_indices]
    lap_pos_enc = (EigVec_sorted[:, 1:pe_dim + 1]).float()
    return lap_pos_enc


def re_features(adj, features, K):
    # size = (N, 1, K+1, d )
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]
    x = features + torch.zeros_like(features)
    x = x.double()
    for i in range(K):
        x = torch.matmul(adj, x)  # Equation (17)
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]
    nodes_features = nodes_features.squeeze()
    # return (N, hops+1, d)
    return nodes_features


class PolynomialDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels


def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(directory+'/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AUPR: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AUPR: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.pdf' % name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
