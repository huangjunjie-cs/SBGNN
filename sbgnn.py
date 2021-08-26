#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: huangjunjie
@file: sbgnn.py
@time: 2021/03/28
"""


import os
import sys
import time
import random
import argparse
import subprocess

from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


from tqdm import tqdm

import logging
# https://docs.python.org/3/howto/logging.html#logging-advanced-tutorial



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--dirpath', default=BASE_DIR, help='Current Dir')
parser.add_argument('--device', type=str, default='cpu', help='Devices')
parser.add_argument('--dataset_name', type=str, default='house1to10-1')
parser.add_argument('--a_emb_size', type=int, default=32, help='Embeding A Size')
parser.add_argument('--b_emb_size', type=int, default=32, help='Embeding B Size')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight Decay')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning Rate')
parser.add_argument('--seed', type=int, default=13, help='Random seed')
parser.add_argument('--epoch', type=int, default=2000, help='Epoch')
parser.add_argument('--gnn_layer_num', type=int, default=2, help='GNN Layer')
parser.add_argument('--batch_size', type=int, default=500, help='Batch Size')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
parser.add_argument('--agg', type=str, default='AttentionAggregator', choices=['AttentionAggregator', 'MeanAggregator'], help='Aggregator')
args = parser.parse_args()



# TODO 

exclude_hyper_params = ['dirpath', 'device']
hyper_params = dict(vars(args))
for exclude_p in exclude_hyper_params:
    del hyper_params[exclude_p]

hyper_params = "~".join([f"{k}-{v}" for k,v in hyper_params.items()])

from torch.utils.tensorboard import SummaryWriter
# https://pytorch.org/docs/stable/tensorboard.html
tb_writer = SummaryWriter(comment=hyper_params)


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
# setup seed
setup_seed(args.seed)

from common import DATA_EMB_DIC

# args.device = 'cpu'
args.device = torch.device(args.device)

class MeanAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(MeanAggregator, self).__init__()

        self.out_mlp_layer = nn.Sequential(
            nn.Linear(b_dim, b_dim)
        )

    def forward(self, edge_dic_list: dict, feature_a, feature_b, node_num_a, node_num_b):

        edges = []
        for node in range(node_num_a):
            neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
            a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
            edges.append(np.concatenate([a, neighs], axis=1))

        edges = np.vstack(edges)
        edges = torch.LongTensor(edges).to(args.device)
        matrix = torch.sparse_coo_tensor(edges.t(), torch.ones(edges.shape[0]), torch.Size([node_num_a, node_num_b]), device=args.device)
        row_sum = torch.spmm(matrix, torch.ones(size=(node_num_b, 1)).to(args.device))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(args.device), row_sum)

        new_emb = feature_b
        new_emb = self.out_mlp_layer(new_emb)
        output_emb = torch.spmm(matrix, new_emb)
        output_emb = output_emb.div(row_sum)

        return output_emb


class AttentionAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(AttentionAggregator, self).__init__()

        self.out_mlp_layer = nn.Sequential(
            nn.Linear(b_dim, b_dim),
        )

        self.a = nn.Parameter(torch.FloatTensor(a_dim + b_dim, 1))
        nn.init.kaiming_normal_(self.a.data)

    def forward(self, edge_dic_list: dict, feature_a, feature_b, node_num_a, node_num_b):

        edges = []
        for node in range(node_num_a):
            neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
            a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
            edges.append(np.concatenate([a, neighs], axis=1))

        edges = np.vstack(edges)
        edges = torch.LongTensor(edges).to(args.device)
        
        new_emb = feature_b
        new_emb = self.out_mlp_layer(new_emb)

        edge_h_2 = torch.cat([feature_a[edges[:, 0]], new_emb[edges[:, 1]] ], dim=1)
        edges_h = torch.exp(F.elu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), 0.1))
        
        matrix = torch.sparse_coo_tensor(edges.t(), edges_h[:, 0], torch.Size([node_num_a, node_num_b]), device=args.device)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(node_num_b, 1)).to(args.device))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(args.device), row_sum)

        output_emb = torch.sparse.mm(matrix, new_emb)
        output_emb = output_emb.div(row_sum)
        return output_emb



class SBGNNLayer(nn.Module):
    def __init__(self, edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,\
                    edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg, \
        dataset_name=args.dataset_name, emb_size_a=32, emb_size_b=32, aggregator=MeanAggregator):
        super(SBGNNLayer, self).__init__()
        # 
        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]
        
        # self.feature_a = feature_a
        # self.feature_b = feature_b
        self.edgelist_a_b_pos, self.edgelist_a_b_neg, self.edgelist_b_a_pos, self.edgelist_b_a_neg = \
            edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg
        self.edgelist_a_a_pos, self.edgelist_a_a_neg, self.edgelist_b_b_pos, self.edgelist_b_b_neg = \
            edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg

        self.agg_a_from_b_pos = aggregator(emb_size_b, emb_size_a)
        self.agg_a_from_b_neg = aggregator(emb_size_b, emb_size_a)
        self.agg_a_from_a_pos = aggregator(emb_size_a, emb_size_a)
        self.agg_a_from_a_neg = aggregator(emb_size_a, emb_size_a)

        self.agg_b_from_a_pos = aggregator(emb_size_a, emb_size_b)
        self.agg_b_from_a_neg = aggregator(emb_size_a, emb_size_b)
        self.agg_b_from_b_pos = aggregator(emb_size_b, emb_size_b)
        self.agg_b_from_b_neg = aggregator(emb_size_b, emb_size_b)

        self.update_func = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(emb_size_a * 5, emb_size_a * 2),
            nn.PReLU(),
            nn.Linear(emb_size_b * 2, emb_size_b)
            
        )
       


    def forward(self, feature_a, feature_b):
        # assert feature_a.size()[0] == self.set_a_num, 'set_b_num error'
        # assert feature_b.size()[0] == self.set_b_num, 'set_b_num error'

        node_num_a, node_num_b = self.set_a_num, self.set_b_num

        m_a_from_b_pos = self.agg_a_from_b_pos(self.edgelist_a_b_pos, feature_a, feature_b, node_num_a, node_num_b)
        m_a_from_b_neg = self.agg_a_from_b_neg(self.edgelist_a_b_neg, feature_a, feature_b, node_num_a, node_num_b)
        m_a_from_a_pos = self.agg_a_from_a_pos(self.edgelist_a_a_pos, feature_a, feature_a, node_num_a, node_num_a)
        m_a_from_a_neg = self.agg_a_from_a_neg(self.edgelist_a_a_neg, feature_a, feature_a, node_num_a, node_num_a)
        
        new_feature_a = torch.cat([feature_a, m_a_from_b_pos, m_a_from_b_neg, m_a_from_a_pos, m_a_from_a_neg], dim=1)
        new_feature_a = self.update_func(new_feature_a)

        m_b_from_a_pos = self.agg_b_from_a_pos(self.edgelist_b_a_pos, feature_b, feature_a, node_num_b, node_num_a)
        m_b_from_a_neg = self.agg_b_from_a_neg(self.edgelist_b_a_pos, feature_b, feature_a, node_num_b, node_num_a)
        m_b_from_b_pos = self.agg_b_from_b_pos(self.edgelist_b_b_pos, feature_b, feature_b, node_num_b, node_num_b)
        m_b_from_b_neg = self.agg_b_from_b_neg(self.edgelist_b_b_neg, feature_b, feature_b, node_num_b, node_num_b)

        new_feature_b = torch.cat([feature_b, m_b_from_a_pos, m_b_from_a_neg, m_b_from_b_pos, m_b_from_b_neg], dim=1)
        new_feature_b = self.update_func(new_feature_b)

        return new_feature_a, new_feature_b



class SBGNN(nn.Module):
    def __init__(self, edgelists,
                    dataset_name=args.dataset_name, layer_num=1, emb_size_a=32, emb_size_b=32, aggregator=AttentionAggregator):
        super(SBGNN, self).__init__()

        # assert edgelists must compelte
        assert len(edgelists) == 8, 'must 8 edgelists'
        edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,\
                    edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg = edgelists
        
        self.set_a_num, self.set_b_num = DATA_EMB_DIC[dataset_name]

        self.features_a = nn.Embedding(self.set_a_num, emb_size_a)
        self.features_b = nn.Embedding(self.set_b_num, emb_size_b)
        self.features_a.weight.requires_grad = True
        self.features_b.weight.requires_grad = True        
        # features_a = features_a.to(args.device)
        # features_b = features_b.to(args.device)

        self.layers = nn.ModuleList(
            [SBGNNLayer(edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,\
                    edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg, \
                    dataset_name=dataset_name, emb_size_a=32, emb_size_b=32, aggregator=aggregator) for _ in range(layer_num)]
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(emb_size_a * 3, 30),
        #     nn.PReLU(),
        #     nn.Linear(30, 1),
        #     nn.Sigmoid()
        # )
        # def init_weights(m):
        #     if type(m) == nn.Linear:
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0.01)
        # self.apply(init_weights)


    def get_embeddings(self):
        emb_a = self.features_a(torch.arange(self.set_a_num).to(args.device))
        emb_b = self.features_b(torch.arange(self.set_b_num).to(args.device))
        for m in self.layers:
            emb_a, emb_b = m(emb_a, emb_b)
        return emb_a, emb_b

    def forward(self, edge_lists):
        embedding_a, embedding_b = self.get_embeddings()

        #### with mlp
        # emb_concat = torch.cat([embedding_a[edge_lists[:, 0]], embedding_b[edge_lists[:, 1]], embedding_a[edge_lists[:, 0]] * embedding_b[edge_lists[:, 1]] ], dim=1)
        # y = self.mlp(emb_concat).squeeze(-1)
        # return y

        ## without mlp
        y = torch.einsum("ij, ij->i", [embedding_a[edge_lists[:, 0]] , embedding_b[edge_lists[:, 1]] ])
        return torch.sigmoid(y)   

    def loss(self, pred_y, y):
        assert y.min() >= 0, 'must 0~1'
        assert pred_y.size() == y.size(), 'must be same length'
        pos_ratio = y.sum() /  y.size()[0]
        weight = torch.where(y > 0.5, 1./pos_ratio, 1./(1-pos_ratio))
        # weight = torch.where(y > 0.5, (1-pos_ratio), pos_ratio)
        return F.binary_cross_entropy(pred_y, y, weight=weight)


# =========== function
def load_data(dataset_name):
    train_file_path = os.path.join('experiments-data', f'{dataset_name}_training.txt')
    val_file_path = os.path.join('experiments-data', f'{dataset_name}_validation.txt')
    test_file_path = os.path.join('experiments-data', f'{dataset_name}_testing.txt')
    
    train_edgelist = []
    with open(train_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
            train_edgelist.append((a, b, s))

    val_edgelist = []
    with open(val_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
            val_edgelist.append((a, b, s))

    test_edgelist = []
    with open(test_file_path) as f:
        for ind, line in enumerate(f):
            if ind == 0: continue
            a, b, s = map(int, line.split('\t'))
            test_edgelist.append((a, b, s))

    return np.array(train_edgelist), np.array(val_edgelist), np.array(test_edgelist)


# ============= load data
def load_edgelists(edge_lists):
    edgelist_a_b_pos, edgelist_a_b_neg = defaultdict(list), defaultdict(list) 
    edgelist_b_a_pos, edgelist_b_a_neg = defaultdict(list), defaultdict(list)
    edgelist_a_a_pos, edgelist_a_a_neg = defaultdict(list), defaultdict(list)
    edgelist_b_b_pos, edgelist_b_b_neg = defaultdict(list), defaultdict(list)

    for a, b, s in edge_lists:
        if s == 1:
            edgelist_a_b_pos[a].append(b)
            edgelist_b_a_pos[b].append(a)
        elif s== -1:
            edgelist_a_b_neg[a].append(b)
            edgelist_b_a_neg[b].append(a)
        else:
            print(a, b, s)
            raise Exception("s must be -1/1")

    edge_list_a_a = defaultdict(lambda: defaultdict(int))
    edge_list_b_b = defaultdict(lambda: defaultdict(int))
    for a, b, s in edge_lists:
        for b2 in edgelist_a_b_pos[a]:
            edge_list_b_b[b][b2] += 1 * s
        for b2 in edgelist_a_b_neg[a]:
            edge_list_b_b[b][b2] -= 1 * s
        for a2 in edgelist_b_a_pos[b]:
            edge_list_a_a[a][a2] += 1 * s
        for a2 in edgelist_b_a_neg[b]:
            edge_list_a_a[a][a2] -= 1 * s

    for a1 in edge_list_a_a:
        for a2 in edge_list_a_a[a1]:
            v = edge_list_a_a[a1][a2]
            if a1 == a2: continue
            if v > 0:
                edgelist_a_a_pos[a1].append(a2)
            elif v < 0:
                edgelist_a_a_neg[a1].append(a2) 

    for b1 in edge_list_b_b:
        for b2 in edge_list_b_b[b1]:
            v = edge_list_b_b[b1][b2]
            if b1 == b2: continue
            if v > 0:
                edgelist_b_b_pos[b1].append(b2)
            elif v < 0:
                edgelist_b_b_neg[b1].append(b2) 

    return edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,\
                    edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg                 


@torch.no_grad()
def test_and_val(pred_y, y, mode='val', epoch=0):
    preds = pred_y.cpu().numpy()
    y = y.cpu().numpy()

    preds[preds >= 0.5]  = 1
    preds[preds < 0.5] = 0
    test_y = y

    auc = roc_auc_score(test_y, preds)
    f1 = f1_score(test_y, preds)
    macro_f1 = f1_score(test_y, preds, average='macro')
    micro_f1 = f1_score(test_y, preds, average='micro')
    pos_ratio = np.sum(test_y) /  len(test_y)
    res = {
        f'{mode}_auc': auc,
        f'{mode}_f1' : f1,
        f'{mode}_pos_ratio': pos_ratio,
        f'{mode}_epoch': epoch,
        f'{mode}_macro_f1' : macro_f1,
        f'{mode}_micro_f1' : micro_f1,
    }
    for k, v in res.items():
        mode ,_, metric = k.partition('_')
        tb_writer.add_scalar(f'{metric}/{mode}', v, epoch)
    # tb_writer.add_scalar( f'{mode}_auc', auc, epoch)
    # tb_writer.add_scalar( f'{mode}_f1', auc, epoch)
    return res



def run():
    train_edgelist, val_edgelist, test_edgelist  = load_data(args.dataset_name)

    set_a_num, set_b_num = DATA_EMB_DIC[args.dataset_name]
    train_y = np.array([i[-1] for i in train_edgelist])
    val_y   = np.array([i[-1] for i in val_edgelist])
    test_y  = np.array([i[-1] for i in test_edgelist])

    train_y = torch.from_numpy( (train_y + 1)/2 ).float().to(args.device)
    val_y = torch.from_numpy( (val_y + 1)/2 ).float().to(args.device)
    test_y = torch.from_numpy( (test_y + 1)/2 ).float().to(args.device)
    # get edge lists 
    edgelists = load_edgelists(train_edgelist)

    if args.agg == 'MeanAggregator':
        agg = MeanAggregator
    else:
        agg = AttentionAggregator

    model = SBGNN(edgelists, dataset_name=args.dataset_name, layer_num=args.gnn_layer_num, aggregator=agg)
    model = model.to(args.device)

    print(model.train())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    res_best = {'val_auc': 0}
    for epoch in tqdm(range(1, args.epoch + 2)):
        # train
        model.train()
        optimizer.zero_grad()
        pred_y = model(train_edgelist)
        loss = model.loss(pred_y, train_y)
        loss.backward()
        optimizer.step()
        print('loss', loss)
        

        res_cur = {}
        # if epoch % 5 == 0:
        if True:
        # val/test
            model.eval()
            pred_y = model(train_edgelist)
            res = test_and_val(pred_y, train_y, mode='train', epoch=epoch)
            res_cur.update(res)
            pred_val_y = model(val_edgelist)
            res = test_and_val(pred_val_y, val_y, mode='val', epoch=epoch)
            res_cur.update(res)
            pred_test_y = model(test_edgelist)
            res = test_and_val(pred_test_y, test_y, mode='test', epoch=epoch)
            res_cur.update(res)
            if res_cur['val_auc'] > res_best['val_auc']:
                res_best = res_cur
                print(res_best)
    print('Done! Best Results:')
    print(res_best)
    print_list = ['test_auc', 'test_f1', 'test_macro_f1', 'test_micro_f1']
    for i in print_list:
        print(i, res_best[i], end=' ')



def main():
    print(" ".join(sys.argv))
    this_fpath = os.path.abspath(__file__)
    t = subprocess.run(f'cat {this_fpath}', shell=True, stdout=subprocess.PIPE)
    print(str(t.stdout, 'utf-8'))
    print('=' * 20)
    run()

if __name__ == "__main__":
    main()
