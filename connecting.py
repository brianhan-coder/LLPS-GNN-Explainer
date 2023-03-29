from logging import exception
import os
from platform import architecture
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GCNConv
import networkx as nx
import feature_embedding
import PDB2Graph
import GNN_core
import argparse
import random
from os.path import exists
from multiprocessing import Pool
import multiprocessing
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

### load proteins
protein_dataset = './data_base/regulator_PDB.dat'
pdb_path = 'graph_base/all_graph_PDB'

patience = 20
partition_ratio = 0.7
partition_size = 'max'
lr = 0.01
n_epochs = 100 
arch = 'GCN'
# ratio = args.partition_ratio.split(":")
# ratio = [float(entry) for entry in ratio]
ratio = [0.4, 0.3, 0.3]
batch_size = 40
num_layers = 0
hidden_channels = 12
if partition_size != 'max':
    parition_size = int(partition_size)

proteins=[]
graph_labels=[]
with open(protein_dataset, "r") as file:
    content = file.read()
for line in content.splitlines():
    line=np.array(list(line.split(" ")))
    proteins.append(line[0])
    graph_labels.append(int(line[1]))

tmp = list(zip(proteins, graph_labels))
random.seed(40)
random.shuffle(tmp)
proteins, graph_labels = zip(*tmp)
proteins, graph_labels = list(proteins), list(graph_labels)
if partition_size != 'max':
    proteins=proteins[:int(partition_size)]
    graph_labels=graph_labels[:int(partition_size)]
    
for protein_index,my_protein in enumerate(proteins):
    if os.path.exists(str(pdb_path)+'/'+str(my_protein)+".nx"):
        print('protein_index:',protein_index)
        print(str(pdb_path)+'/'+str(my_protein)+".nx")
    
    if protein_index==5:
        break
        
### parallel converting PDB to graphs 
graph_dataset=[]
for protein_index,my_protein in enumerate(proteins):
    if os.path.exists(str(pdb_path)+'/'+str(my_protein)+".nx"):
        G = nx.read_gpickle(str(pdb_path)+'/'+str(my_protein)+".nx")
#         print(my_protein)
#         print(G)
        graph_dataset.append(G)

data = graph_dataset[1]
# print(graph_dataset[1].edge_attr)
print()
print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
