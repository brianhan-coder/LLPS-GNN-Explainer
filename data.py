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
