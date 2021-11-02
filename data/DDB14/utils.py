import torch
import os
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import random
import itertools
import multiprocessing as mp
import time

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def create_graph(entity2id, triples, relation):
    G = nx.MultiDiGraph()
    num = len(entity2id)
    for node in range(num):
        G.add_node(node)
    for triple in triples:
        u, r, v = triple
        if r == relation:
            judge = nx.has_path(G, source=u, target=v)
            if judge:
                print(relation, 'found!')
            G.add_edge(u, v)
    return G

data_path = './'

with open(os.path.join(data_path, 'entities.dict')) as fin:
    entity2id = dict()
    id2entity = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)
        id2entity[int(eid)] = entity
    nentity = len(entity2id)

with open(os.path.join(data_path, 'relations.dict')) as fin:
    relation2id = dict()
    id2relation = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)
        id2relation[int(rid)] = relation
'''
train_triples = read_triple('./train.txt', entity2id, relation2id)
for i in range(14):
    train_graph = create_graph(entity2id, train_triples, 3)
'''
lines = []
with open(os.path.join(data_path, 'class_test_easy.txt')) as fin:
    for line in fin:
        lhs, rel, rhs, gap = line.strip().split('\t')
        lines.append(id2entity[int(lhs)]+'\t'+id2relation[int(rel)]+'\t'+id2entity[int(rhs)]+'\t'+gap+'\n')
with open("RotH/class_test_easy", "w") as f:
    f.writelines(lines)