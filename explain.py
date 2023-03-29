import GNN_core

from importlib import reload
reload(GNN_core)
# load model
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if arch == 'GCN':
    best_model = GNN_core.GCN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
if arch == 'GNN':
    best_model = GNN_core.GNN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)
if arch == 'GTN':
    best_model = GNN_core.GTN(hidden_channels,num_node_features=num_node_features,num_classes=num_classes,num_layers=num_layers)

save_path = os.path.join('.','{}_best_model.pth'.format(arch))
best_model.load_state_dict(torch.load(save_path))
model.to(device)
model.eval()

from torch_geometric.data import Data
from torch_geometric.explain import Explainer, PGExplainer, Explanation

# data = Data(...)  # A homogeneous graph data object.

explainer = Explainer(
    model=best_model,
#     model=model,
    algorithm=PGExplainer(epochs=50, lr=0.003),
    explanation_type='phenomenon',
#     node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
#         mode='binary_classification',
        mode='multiclass_classification',
        task_level='graph',
        return_type='raw',  # Model returns log probabilities.
    ),
)

# data=graph_dataset[1]
ex_loader = DataLoader(train_dataset[:], batch_size=1, shuffle=False)
for i, data in enumerate(ex_loader):
    print(i,data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    if i==2:
        break
       
for epoch in range(50):
    print('Current epoch', epoch)
    for i, batch in enumerate(ex_loader):
#         print('Current ID:',i)
#         print(data)
#         print(f'Number of nodes: {data.num_nodes}')
#         print(f'Number of edges: {data.num_edges}')
#         print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
#         print('Labels:',data.y)
#         print('Batch:',data.batch.shape)
    
        loss = explainer.algorithm.train(
            epoch, best_model, batch.x, batch.edge_index, target=batch.y, batch=batch.batch)
        
        if i == 10:
            break

# Generate explanation for the node at index `0`:
for i, data in enumerate(ex_loader):
    if data.x.shape[0] > 500:
        continue
        
print('Current ID:',i)
print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print('Labels:',data.y)
print('Batch:',data.batch.shape)
    
explanation = explainer(x=data.x, edge_index=data.edge_index, target = data.y,batch = data.batch#index=0)
print(explanation)
print('explanation target:',explanation.target)
#     print(explanation.edge_mask)
#     print(explanation.node_mask)
explanation.visualize_graph()
print('Sub-graph:')
exp = Explanation(x=data.x, 
edge_index=data.edge_index,
#                   node_mask=explanation.node_mask,
edge_mask=explanation.edge_mask)
print(explanation.edge_mask.shape)
print(torch.sum(explanation.edge_mask))

subgraph = exp.get_explanation_subgraph()
print(torch.sum(subgraph.edge_mask))
subgraph.visualize_graph()
    
complement = exp.get_complement_subgraph()
print(complement)
complement.visualize_graph()
print()
    
if i >= 1:
    break
        
exp = Explanation(x=data.x, 
                  edge_index=data.edge_index,
                  node_mask=explanation.node_mask,
                  edge_mask=explanation.edge_mask)
print(explanation.edge_mask.shape)
print(torch.sum(explanation.edge_mask))

print(explanation.edge_mask[:100])

subgraph = exp.get_explanation_subgraph()
print(subgraph)
print(torch.sum(subgraph.edge_mask))
print(exp.get_complement_subgraph())
# subgraph.visualize_graph()


import torch
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt

print(data.edge_index.shape)

adj_matrix = to_dense_adj(data.edge_index)
print(adj_matrix.shape)

a = adj_matrix[0,...]
# a = adj_matrix[0,:100,:100] 
fig = plt.figure(figsize=(8,8))
# plt.xticks([])
# plt.yticks([])
plt.imshow(a)
plt.colorbar()
plt.show()

data.edge_index[:,:14]

