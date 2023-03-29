# balance the data
print('____')
print('Before balance')
print('length graph_dataset',len(graph_dataset))
sum_i = 0
for i in graph_dataset:
    sum_i = sum_i + i.y

print('#Label1 and ration:',sum_i,sum_i/len(graph_dataset))
print('____')

### train test partition
graph_dataset=GNN_core.balance_dataset(graph_dataset)
print('____')
print('After balance')
print('length graph_dataset',len(graph_dataset))
sum_i = 0
for i in graph_dataset:
    sum_i = sum_i + i.y

print('#Label1 and ration:',sum_i,sum_i/len(graph_dataset))
GNN_core.get_info_dataset(graph_dataset,verbose=True)
print('____')

#train_test_partition=int(partition_ratio*len(graph_dataset))
assert(ratio[0]+ratio[1]+ratio[2]==1)
part1 = int(len(graph_dataset)*ratio[0])
part2 = part1 + int(len(graph_dataset)*ratio[1])
part3 = part2 + int(len(graph_dataset)*ratio[2])

train_dataset = graph_dataset[:part1]
test_dataset = graph_dataset[part1:part2]
val_dataset = graph_dataset[part2:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print(f'Number of val graphs: {len(val_dataset)}')

print('____')
print('Length test_dataset',len(test_dataset))
sum_i = 0
for i in test_dataset:
    sum_i = sum_i + i.y

print('#Label1 and ration:',sum_i,sum_i/len(test_dataset))
GNN_core.get_info_dataset(test_dataset,verbose=True)
print('____')
print('Data info for checking randomness:')
for i, data in enumerate(test_dataset):
    print(data)
    if i == 10:
        break

### mini-batching of graphs, adjacency matrices are stacked in a diagonal fashion. 
### Batching multiple graphs into a single giant graph
from torch_geometric.loader import DataLoader
print(len(train_dataset))
print(batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

print('For each batch in train_loader:')
for i, data in enumerate(train_loader):
    print(i)
    print(data.ptr)
    print(data)
    print(data.y)
#     print(data.batch[:10000])
    print()

import GNN_core

from importlib import reload
reload(GNN_core)

from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
### core GNN 
num_node_features=len(graph_dataset[0].x[0])
num_classes=2

if arch == 'GCN':
    model = GNN_core.GCN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
if arch == 'GNN':
    model = GNN_core.GNN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
if arch == 'GTN':
    model = GNN_core.GTN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)

print(type(model))  
# for i, layer in enumerate(model.children()):
#     print(i)
#     print(layer)
#     print(isinstance(layer, GCNConv))
#     dic = layer.state_dict()
#     print(dic)
# #     for k in dic:
# #         print(dic[k])

### randomly initialize GCNConv model parameters
for layer in model.children():
    if isinstance(layer, GCNConv):
        dic = layer.state_dict()
        for k in dic:
            dic[k] = torch.randn(dic[k].size())
        layer.load_state_dict(dic)
        del(dic)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.0001)
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss()

best_val_acc = 0
best_val_epoch = 0
best_model=None
